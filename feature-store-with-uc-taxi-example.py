# Databricks notebook source
# MAGIC %md
# MAGIC # Unity Catalogでの特徴量エンジニアリングの高度な例
# MAGIC
# MAGIC このノートブックは、Unity Catalogで特徴量エンジニアリングを使用してNYCイエローキャブの運賃を予測するモデルを作成する方法を示しています。以下のステップが含まれます：
# MAGIC
# MAGIC - Unity Catalogで直接時系列特徴量を計算して書き込む。
# MAGIC - これらの特徴量を使用して運賃を予測するモデルをトレーニングする。
# MAGIC - 既存の特徴量を使用して新しいデータバッチでそのモデルを評価する。
# MAGIC
# MAGIC ## 要件
# MAGIC - Databricks Runtime 13.3 LTS for Machine Learning以上
# MAGIC   - Databricks Runtime for Machine Learningにアクセスできない場合は、このノートブックをDatabricks Runtime 13.3 LTS以上で実行できます。その場合、このノートブックの最初に`%pip install databricks-feature-engineering`を実行してください。

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_flow_v3.png"/>

# COMMAND ----------

CATALOG_NAME = "users"
SCHEMA_NAME = "takaaki_yayoi"

# COMMAND ----------

# MAGIC %md ## 特徴量の計算

# COMMAND ----------

# MAGIC %md #### 特徴量の計算に使用する生データの読み込み
# MAGIC
# MAGIC `nyc-taxi-tiny` データセットを読み込みます。これは、以下の変換を適用して、完全な [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) から `/databricks-datasets/nyctaxi-with-zipcodes/subsampled` に生成されました。
# MAGIC
# MAGIC 1. 緯度と経度の座標を ZIP コードに変換する UDF を適用し、DataFrame に ZIP コード列を追加します。
# MAGIC 1. Spark `DataFrame` API の `.sample()` メソッドを使用して、日付範囲クエリに基づいてデータセットを小さなデータセットにサブサンプリングします。
# MAGIC 1. 特定の列の名前を変更し、不要な列を削除します。

# COMMAND ----------

raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC タクシー料金の取引データから、乗車および降車の ZIP コードに基づいて2つのグループの特徴量を計算します。
# MAGIC
# MAGIC #### 乗車特徴量
# MAGIC 1. 旅行回数（時間ウィンドウ = 1時間、スライディングウィンドウ = 15分）
# MAGIC 1. 平均運賃額（時間ウィンドウ = 1時間、スライディングウィンドウ = 15分）
# MAGIC
# MAGIC #### 降車特徴量
# MAGIC 1. 旅行回数（時間ウィンドウ = 30分）
# MAGIC 1. 旅行が週末に終了するかどうか（Pythonコードを使用したカスタム特徴量）
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_computation_v5.png"/>

# COMMAND ----------

# MAGIC %md ### ヘルパー関数

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone


@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = 土曜日, 6 = 日曜日


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df

# COMMAND ----------

# MAGIC %md ### データサイエンティストのカスタムコードによる特徴量の計算

# COMMAND ----------

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    pickup_features フィーチャーグループを計算します。
    フィーチャーを特定の時間範囲に制限するには、ts_column、start_date、および/または end_date を kwargs として渡します。
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1時間のウィンドウ、15分ごとにスライド
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).cast("timestamp").alias("ts"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features


def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    dropoff_features フィーチャーグループを計算します。
    フィーチャーを特定の時間範囲に制限するには、ts_column、start_date、および/または end_date を kwargs として渡します。
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).cast("timestamp").alias("ts"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features

# COMMAND ----------

from datetime import datetime

pickup_features = pickup_features_fn(
    df=raw_data,
    ts_column="tpep_pickup_datetime",
    start_date=datetime(2016, 1, 1),
    end_date=datetime(2016, 1, 31),
)
dropoff_features = dropoff_features_fn(
    df=raw_data,
    ts_column="tpep_dropoff_datetime",
    start_date=datetime(2016, 1, 1),
    end_date=datetime(2016, 1, 31),
)

# COMMAND ----------

display(pickup_features)

# COMMAND ----------

display(dropoff_features)

# COMMAND ----------

# MAGIC %md ### Unity Catalogで新しい時系列特徴量テーブルを作成する

# COMMAND ----------

# MAGIC %md まず、新しいカタログを作成するか、既存のカタログを再利用して、特徴量テーブルを格納するスキーマを作成します。
# MAGIC - 新しいカタログを作成するには、メタストアに対する `CREATE CATALOG` 権限が必要です。
# MAGIC - 既存のカタログを使用するには、カタログに対する `USE CATALOG` 権限が必要です。
# MAGIC - カタログに新しいスキーマを作成するには、カタログに対する `CREATE SCHEMA` 権限が必要です。

# COMMAND ----------

# カタログとスキーマを指定。存在しない場合には作成してください。
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md 次に、主キー制約を使用してUnity Catalogに時系列特徴量テーブルを作成します。
# MAGIC
# MAGIC `CREATE TABLE` SQL構文を使用してUnity Catalogに直接テーブルを作成できます。主キー制約を使用して主キー列を指定します。時系列テーブルの場合、`TIMESERIES`を使用して時系列列を注釈します（[AWS](https://docs.databricks.com/ja/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint.html)｜[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint)｜[GCP](https://docs.databricks.com/gcp/ja/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint)）。
# MAGIC
# MAGIC 時系列の列は`TimestampType`または`DateType`でなければなりません。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS trip_pickup_time_series_features(
# MAGIC   zip INT NOT NULL,
# MAGIC   ts TIMESTAMP NOT NULL,
# MAGIC   mean_fare_window_1h_pickup_zip FLOAT,
# MAGIC   count_trips_window_1h_pickup_zip INT,
# MAGIC   CONSTRAINT trip_pickup_time_series_features_pk PRIMARY KEY (zip, ts TIMESERIES)
# MAGIC )
# MAGIC COMMENT "タクシー料金。乗車時系列の特徴量。";

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS trip_dropoff_time_series_features(
# MAGIC   zip INT NOT NULL,
# MAGIC   ts TIMESTAMP NOT NULL,
# MAGIC   count_trips_window_30m_dropoff_zip INT,
# MAGIC   dropoff_is_weekend INT,
# MAGIC   CONSTRAINT trip_dropoff_time_series_features_pk PRIMARY KEY (zip, ts TIMESERIES)
# MAGIC )
# MAGIC COMMENT "タクシー料金。降車時系列の特徴量。";

# COMMAND ----------

# MAGIC %md ### 初期特徴量をUnity Catalogの特徴量テーブルに書き込む

# COMMAND ----------

# MAGIC %md Feature Engineeringクライアントのインスタンスを作成します。

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC `write_table` APIを使用して、特徴量をUnity Catalogの特徴量テーブルに書き込みます。
# MAGIC
# MAGIC 時系列特徴量テーブルに書き込むには、DataFrameに時系列列として指定する列が含まれている必要があります。

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")
fe.write_table(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_pickup_time_series_features",
    df=pickup_features
)
fe.write_table(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_dropoff_time_series_features",
    df=dropoff_features
)

# COMMAND ----------

# MAGIC %md ## 特徴量の更新
# MAGIC
# MAGIC `write_table` 関数を使用して特徴量テーブルの値を更新します。

# COMMAND ----------

display(raw_data)

# COMMAND ----------

# 新しいバッチのpickup_featuresフィーチャーグループを計算する。
new_pickup_features = pickup_features_fn(
    df=raw_data,
    ts_column="tpep_pickup_datetime",
    start_date=datetime(2016, 2, 1),
    end_date=datetime(2016, 2, 29),
)
# 新しいpickup features DataFrameをフィーチャーテーブルに書き込む
fe.write_table(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_pickup_time_series_features",
    df=new_pickup_features,
    mode="merge",
)

# 新しいバッチのdropoff_featuresフィーチャーグループを計算する。
new_dropoff_features = dropoff_features_fn(
    df=raw_data,
    ts_column="tpep_dropoff_datetime",
    start_date=datetime(2016, 2, 1),
    end_date=datetime(2016, 2, 29),
)
# 新しいdropoff features DataFrameをフィーチャーテーブルに書き込む
fe.write_table(
    name=f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_dropoff_time_series_features",
    df=new_dropoff_features,
    mode="merge",
)

# COMMAND ----------

# MAGIC %md 書き込み時には、`merge` モードがサポートされています。
# MAGIC
# MAGIC     fe.write_table(
# MAGIC       name="ml.taxi_example.trip_pickup_time_series_features",
# MAGIC       df=new_pickup_features,
# MAGIC       mode="merge",
# MAGIC     )
# MAGIC
# MAGIC データは、`df.isStreaming` が `True` に設定されているデータフレームを渡すことで、特徴量テーブルにストリーミングすることもできます:
# MAGIC
# MAGIC     fe.write_table(
# MAGIC       name="ml.taxi_example.trip_pickup_time_series_features",
# MAGIC       df=streaming_pickup_features,
# MAGIC       mode="merge",
# MAGIC     )
# MAGIC
# MAGIC Databricks Jobsを使用してノートブックを定期的にスケジュールし、特徴量を更新することができます ([AWS](https://docs.databricks.com/aws/ja/jobs)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/jobs/)|[GCP](https://docs.databricks.com/gcp/ja/jobs))。

# COMMAND ----------

# MAGIC %md アナリストは、例えば次のようにSQLを使用してUnity Catalogの特徴量テーブルと対話できます:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   SUM(count_trips_window_30m_dropoff_zip) AS num_rides,
# MAGIC   dropoff_is_weekend
# MAGIC FROM
# MAGIC   trip_dropoff_time_series_features
# MAGIC WHERE
# MAGIC   dropoff_is_weekend IS NOT NULL
# MAGIC GROUP BY
# MAGIC   dropoff_is_weekend;

# COMMAND ----------

# MAGIC %md ## 特徴量の検索と発見

# COMMAND ----------

# MAGIC %md
# MAGIC Unity Catalogの<a href="#feature-store/feature-store" target="_blank">Features UI</a>で特徴量テーブルを発見できます。"trip_pickup_time_series_features"または"trip_dropoff_time_series_features"で検索し、テーブル名をクリックして、カタログエクスプローラーUIでテーブルスキーマ、メタデータ、系統などの詳細を確認できます。特徴量テーブルの説明も編集できます。特徴量の発見と系統追跡の詳細については、([AWS](https://docs.databricks.com/aws/ja/machine-learning/feature-store/uc/ui-uc)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/feature-store/uc/ui-uc)|[GCP](https://docs.databricks.com/gcp/ja/machine-learning/feature-store/uc/ui-uc))を参照してください。
# MAGIC
# MAGIC カタログエクスプローラーUIで特徴量テーブルの権限も設定できます。詳細については、([AWS](https://docs.databricks.com/aws/ja/data-governance/unity-catalog/manage-privileges)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/data-governance/unity-catalog/manage-privileges/)|[GCP](https://docs.databricks.com/gcp/ja/data-governance/unity-catalog/manage-privileges))を参照してください。

# COMMAND ----------

# MAGIC %md ## モデルのトレーニング
# MAGIC
# MAGIC このセクションでは、ポイントインタイムルックアップを使用して時系列のpickupおよびdropoff特徴量テーブルを使用してトレーニングセットを作成し、トレーニングセットを使用してモデルをトレーニングする方法を説明します。タクシー料金を予測するためにLightGBMモデルをトレーニングします。

# COMMAND ----------

# MAGIC %md ### ヘルパー関数

# COMMAND ----------

import mlflow.pyfunc

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# MAGIC %md ### トレーニングデータセットの作成方法の理解
# MAGIC
# MAGIC モデルをトレーニングするためには、トレーニングデータセットを作成する必要があります。トレーニングデータセットは以下で構成されます：
# MAGIC
# MAGIC 1. 生の入力データ
# MAGIC 1. Unity Catalogの特徴量テーブルからの特徴量
# MAGIC
# MAGIC 生の入力データが必要な理由は以下の通りです：
# MAGIC
# MAGIC 1. 主キーおよび時系列カラムは、ポイントインタイムの正確性を持つ特徴量と結合するために使用されます ([AWS](https://docs.databricks.com/aws/ja/machine-learning/feature-store/time-series#create-a-training-set-with-a-time-series-feature-table)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/feature-store/time-series#create-a-training-set-with-a-time-series-feature-table)|[GCP](https://docs.databricks.com/gcp/ja/machine-learning/feature-store/time-series#create-a-training-set-with-a-time-series-feature-table))。
# MAGIC 1. 特徴量テーブルに含まれていない`trip_distance`のような生の特徴量。
# MAGIC 1. モデルのトレーニングに必要な`fare`のような予測ターゲット。
# MAGIC
# MAGIC 以下のビジュアル概要は、生の入力データがUnity Catalogの特徴量と組み合わされてトレーニングデータセットを生成する様子を示しています：
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi-example-feature-lookup-with-pit.png"/>
# MAGIC
# MAGIC これらの概念は、トレーニングデータセット作成のドキュメントでさらに説明されています ([AWS](https://docs.databricks.com/aws/ja/machine-learning/feature-store/train-models-with-feature-store#create-a-training-dataset)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/feature-store/train-models-with-feature-store#create-a-training-dataset)|[GCP](https://docs.databricks.com/gcp/ja/machine-learning/feature-store/train-models-with-feature-store#create-a-training-dataset))。
# MAGIC
# MAGIC 次のセルでは、必要な特徴量ごとに`FeatureLookup`を作成して、Unity Catalogからモデルトレーニング用の特徴量をロードします。
# MAGIC
# MAGIC 時系列特徴量テーブルから特徴量値をポイントインタイムでルックアップするには、特徴量の`FeatureLookup`で`timestamp_lookup_key`を指定する必要があります。これは、時系列特徴量をルックアップするためのタイムスタンプを含むDataFrameカラムの名前を示します。DataFrameの各行に対して、取得される特徴量値は、DataFrameの`timestamp_lookup_key`カラムに指定されたタイムスタンプより前の最新の特徴量値であり、主キーがDataFrameの`lookup_key`カラムの値と一致するものです。一致する特徴量値が存在しない場合は`null`が返されます。

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup
import mlflow

pickup_features_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_pickup_time_series_features"
dropoff_features_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.trip_dropoff_time_series_features"

pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key="tpep_pickup_datetime",
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key="tpep_dropoff_datetime",
    ),
]

# COMMAND ----------

# MAGIC %md ### MLflowクライアントを設定してUnity Catalogのモデルにアクセスする

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC `fe.create_training_set(..)`が呼び出されると、以下のステップが実行されます：
# MAGIC
# MAGIC 1. `TrainingSet`オブジェクトが作成され、モデルのトレーニングに使用する特定の特徴量が特徴量テーブルから選択されます。各特徴量は、以前に作成された`FeatureLookup`によって指定されます。
# MAGIC
# MAGIC 1. 特徴量は各`FeatureLookup`の`lookup_key`に従って生の入力データと結合されます。
# MAGIC
# MAGIC 1. データリーク問題を回避するためにポイントインタイムルックアップが適用されます。`timestamp_lookup_key`に基づいて、最新の特徴量値のみが結合されます。
# MAGIC
# MAGIC その後、`TrainingSet`はトレーニング用のDataFrameに変換されます。このDataFrameには、taxi_dataのカラムと、`FeatureLookups`で指定された特徴量が含まれます。

# COMMAND ----------

# 既存のランを終了します（このノートブックが2回目以降に実行されている場合）
mlflow.end_run()

# モデルをログに記録するために必要なmlflowランを開始します
mlflow.start_run()

# タイムスタンプ列は追加の特徴量エンジニアリングが行われない限り、モデルがデータに過剰適合する可能性が高いため、
# それらを除外してトレーニングを行わないようにします。
exclude_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# 生データと両方の特徴量テーブルからの対応する特徴量をマージしたトレーニングセットを作成します
training_set = fe.create_training_set(
    df=raw_data,
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
    label="fare_amount",
    exclude_columns=exclude_columns,
)

# sklearnでモデルをトレーニングするために渡すことができるデータフレームにTrainingSetをロードします
training_df = training_set.load_df()

# COMMAND ----------

# トレーニングデータフレームを表示します。これには、生の入力データと特徴量テーブルからの特徴量（例：`dropoff_is_weekend`）の両方が含まれています。
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC `TrainingSet.to_df`で返されたデータに対してLightGBMモデルをトレーニングし、`FeatureEngineeringClient.log_model`でモデルをログします。モデルは特徴量メタデータと共にパッケージ化されます。

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# 特徴量とラベルを含む列を取得します
features_and_label = training_df.columns

# トレーニングのためにデータをPandas配列に収集します
data = training_df.toPandas()[features_and_label]

# データをトレーニングセットとテストセットに分割します
train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

# MLflowの自動ロギングを有効にします
mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

# LightGBMのパラメータを設定します
param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# LightGBMモデルをトレーニングします
model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------

# トレーニング済みモデルをMLflowでログし、特徴量ルックアップ情報と共にパッケージ化します。
fe.log_model(
    model=model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.taxi_example_fare_time_series_packaged",
)

# COMMAND ----------

# MAGIC %md ### カタログエクスプローラーでモデルのリネージを確認
# MAGIC カタログエクスプローラーのテーブル詳細ページにアクセスします。「依存関係」タブに移動し、「リネージグラフを見る」をクリックします。特徴量テーブルに下流のモデルリネージが表示されます。

# COMMAND ----------

# MAGIC %md ### カスタムPyFuncモデルの構築とログ
# MAGIC
# MAGIC モデルに前処理や後処理のコードを追加し、バッチ推論で処理された予測を生成するには、これらのメソッドをカプセル化するカスタムPyFunc MLflowモデルを構築できます。次のセルは、モデルからの数値予測に基づいて文字列出力を返す例を示しています。

# COMMAND ----------

class fareClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model):
        self.model = trained_model

    def preprocess_result(self, model_input):
        return model_input

    def postprocess_result(self, results):
        """予測結果を後処理します。
        運賃の範囲を作成し、予測された範囲を返します。"""

        return [
            "$0 - $9.99" if result < 10 else "$10 - $19.99" if result < 20 else " > $20"
            for result in results
        ]

    def predict(self, context, model_input):
        processed_df = self.preprocess_result(model_input.copy())
        results = self.model.predict(processed_df)
        return self.postprocess_result(results)


pyfunc_model = fareClassifier(model)

# 現在のMLflowランを終了し、新しいpyfuncモデルをログするために新しいランを開始します
mlflow.end_run()

with mlflow.start_run() as run:
    fe.log_model(
        model=pyfunc_model,
        artifact_path="pyfunc_packaged_model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.pyfunc_taxi_fare_time_series_packaged",
    )

# COMMAND ----------

# MAGIC %md ## スコアリング: バッチ推論

# COMMAND ----------

# MAGIC %md 別のデータサイエンティストがこのモデルを別のデータバッチに適用したいとします。

# COMMAND ----------

# MAGIC %md 推論に使用するデータを表示し、予測ターゲットである `fare_amount` 列を強調表示するように並べ替えます。

# COMMAND ----------

cols = [
    "fare_amount",
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]
new_taxi_data = raw_data.select(cols)
display(new_taxi_data)

# COMMAND ----------

# MAGIC %md
# MAGIC `score_batch` API を使用して、Unity Catalog の Feature Engineering から必要な特徴量を取得し、データのバッチでモデルを評価します。
# MAGIC
# MAGIC 時系列特徴量テーブルからの特徴量でトレーニングされたモデルをスコアリングする場合、トレーニング中にモデルにパッケージ化されたメタデータを使用して、適切な特徴量がポイントインタイムルックアップで取得されます。`FeatureEngineeringClient.score_batch` に提供する DataFrame には、`FeatureEngineeringClient.create_training_set` に提供された FeatureLookup の `timestamp_lookup_key` と同じ名前とデータ型のタイムスタンプ列が含まれている必要があります。

# COMMAND ----------

# モデルURIを取得
latest_model_version = get_latest_model_version(
    f"{CATALOG_NAME}.{SCHEMA_NAME}.taxi_example_fare_time_series_packaged"
)
model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.taxi_example_fare_time_series_packaged/{latest_model_version}"

# score_batchを呼び出してモデルから予測を取得
with_predictions = fe.score_batch(model_uri=model_uri, df=new_taxi_data)

# COMMAND ----------

# MAGIC %md ログに記録された PyFunc モデルを使用してスコアリングするには:

# COMMAND ----------

latest_pyfunc_version = get_latest_model_version(
    f"{CATALOG_NAME}.{SCHEMA_NAME}.pyfunc_taxi_fare_time_series_packaged"
)
pyfunc_model_uri = (
    f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.pyfunc_taxi_fare_time_series_packaged/{latest_pyfunc_version}"
)

pyfunc_predictions = fe.score_batch(
    model_uri=pyfunc_model_uri, df=new_taxi_data, result_type="string"
)

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi-example-score-batch-with-pit.png"/>

# COMMAND ----------

# MAGIC %md ### タクシー料金予測を表示
# MAGIC
# MAGIC このコードは列を並べ替えて、タクシー料金の予測を最初の列に表示します。`predicted_fare_amount` が実際の `fare_amount` と大まかに一致することに注意してください。ただし、モデルの精度を向上させるには、より多くのデータと特徴エンジニアリングが必要です。

# COMMAND ----------

import pyspark.sql.functions as func

cols = [
    "prediction",
    "fare_amount",
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "mean_fare_window_1h_pickup_zip",
    "count_trips_window_1h_pickup_zip",
    "count_trips_window_30m_dropoff_zip",
    "dropoff_is_weekend",
]

with_predictions_reordered = (
    with_predictions.select(
        cols,
    )
    .withColumnRenamed(
        "prediction",
        "predicted_fare_amount",
    )
    .withColumn(
        "predicted_fare_amount",
        func.round("predicted_fare_amount", 2),
    )
)

display(with_predictions_reordered)

# COMMAND ----------

# MAGIC %md ### PyFunc 予測を表示

# COMMAND ----------

display(pyfunc_predictions.select("fare_amount", "prediction"))

# COMMAND ----------

# MAGIC %md ## 次のステップ
# MAGIC
# MAGIC 1. <a href="#feature-store/feature-store">Features UI</a> でこの例で作成された特徴量テーブルを探索します。
# MAGIC 1. 特徴テーブルをオンラインストアに公開します ([AWS](https://docs.databricks.com/aws/ja/machine-learning/feature-store/publish-features)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/feature-store/publish-features))。
# MAGIC 1. Unity Catalog でモデルを Model Serving にデプロイします ([AWS](https://docs.databricks.com/aws/ja/machine-learning/model-serving)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/model-serving/))。
# MAGIC 1. このノートブックを自分のデータに適用し、独自の特徴量テーブルを作成します。
