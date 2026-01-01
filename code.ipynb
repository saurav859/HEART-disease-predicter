{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f7cd99a0",
        "outputId": "25ce8005-44fe-463c-b5a2-0ac06027a0e2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id  age     sex    dataset               cp  trestbps   chol    fbs  \\\n",
            "0   1   63    Male  Cleveland   typical angina     145.0  233.0   True   \n",
            "1   2   67    Male  Cleveland     asymptomatic     160.0  286.0  False   \n",
            "2   3   67    Male  Cleveland     asymptomatic     120.0  229.0  False   \n",
            "3   4   37    Male  Cleveland      non-anginal     130.0  250.0  False   \n",
            "4   5   41  Female  Cleveland  atypical angina     130.0  204.0  False   \n",
            "\n",
            "          restecg  thalch  exang  oldpeak        slope   ca  \\\n",
            "0  lv hypertrophy   150.0  False      2.3  downsloping  0.0   \n",
            "1  lv hypertrophy   108.0   True      1.5         flat  3.0   \n",
            "2  lv hypertrophy   129.0   True      2.6         flat  2.0   \n",
            "3          normal   187.0  False      3.5  downsloping  0.0   \n",
            "4  lv hypertrophy   172.0  False      1.4    upsloping  0.0   \n",
            "\n",
            "                thal  num  \n",
            "0       fixed defect    0  \n",
            "1             normal    2  \n",
            "2  reversable defect    1  \n",
            "3             normal    0  \n",
            "4             normal    0  \n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('/content/heart_disease_uci.csv')\n",
        "print(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c1ff00e4",
        "outputId": "e12defb2-b063-41ce-c4c3-cc44d47c976b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DataFrame Head:\n",
            "   id  age     sex    dataset               cp  trestbps   chol    fbs  \\\n",
            "0   1   63    Male  Cleveland   typical angina     145.0  233.0   True   \n",
            "1   2   67    Male  Cleveland     asymptomatic     160.0  286.0  False   \n",
            "2   3   67    Male  Cleveland     asymptomatic     120.0  229.0  False   \n",
            "3   4   37    Male  Cleveland      non-anginal     130.0  250.0  False   \n",
            "4   5   41  Female  Cleveland  atypical angina     130.0  204.0  False   \n",
            "\n",
            "          restecg  thalch  exang  oldpeak        slope   ca  \\\n",
            "0  lv hypertrophy   150.0  False      2.3  downsloping  0.0   \n",
            "1  lv hypertrophy   108.0   True      1.5         flat  3.0   \n",
            "2  lv hypertrophy   129.0   True      2.6         flat  2.0   \n",
            "3          normal   187.0  False      3.5  downsloping  0.0   \n",
            "4  lv hypertrophy   172.0  False      1.4    upsloping  0.0   \n",
            "\n",
            "                thal  num  \n",
            "0       fixed defect    0  \n",
            "1             normal    2  \n",
            "2  reversable defect    1  \n",
            "3             normal    0  \n",
            "4             normal    0  \n",
            "\n",
            "DataFrame Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 920 entries, 0 to 919\n",
            "Data columns (total 16 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   id        920 non-null    int64  \n",
            " 1   age       920 non-null    int64  \n",
            " 2   sex       920 non-null    object \n",
            " 3   dataset   920 non-null    object \n",
            " 4   cp        920 non-null    object \n",
            " 5   trestbps  861 non-null    float64\n",
            " 6   chol      890 non-null    float64\n",
            " 7   fbs       830 non-null    object \n",
            " 8   restecg   918 non-null    object \n",
            " 9   thalch    865 non-null    float64\n",
            " 10  exang     865 non-null    object \n",
            " 11  oldpeak   858 non-null    float64\n",
            " 12  slope     611 non-null    object \n",
            " 13  ca        309 non-null    float64\n",
            " 14  thal      434 non-null    object \n",
            " 15  num       920 non-null    int64  \n",
            "dtypes: float64(5), int64(3), object(8)\n",
            "memory usage: 115.1+ KB\n",
            "\n",
            "Descriptive Statistics:\n",
            "               id         age    trestbps        chol      thalch     oldpeak  \\\n",
            "count  920.000000  920.000000  861.000000  890.000000  865.000000  858.000000   \n",
            "mean   460.500000   53.510870  132.132404  199.130337  137.545665    0.878788   \n",
            "std    265.725422    9.424685   19.066070  110.780810   25.926276    1.091226   \n",
            "min      1.000000   28.000000    0.000000    0.000000   60.000000   -2.600000   \n",
            "25%    230.750000   47.000000  120.000000  175.000000  120.000000    0.000000   \n",
            "50%    460.500000   54.000000  130.000000  223.000000  140.000000    0.500000   \n",
            "75%    690.250000   60.000000  140.000000  268.000000  157.000000    1.500000   \n",
            "max    920.000000   77.000000  200.000000  603.000000  202.000000    6.200000   \n",
            "\n",
            "               ca         num  \n",
            "count  309.000000  920.000000  \n",
            "mean     0.676375    0.995652  \n",
            "std      0.935653    1.142693  \n",
            "min      0.000000    0.000000  \n",
            "25%      0.000000    0.000000  \n",
            "50%      0.000000    1.000000  \n",
            "75%      1.000000    2.000000  \n",
            "max      3.000000    4.000000  \n"
          ]
        }
      ],
      "source": [
        "print(\"DataFrame Head:\")\n",
        "print(df.head())\n",
        "\n",
        "print(\"\\nDataFrame Info:\")\n",
        "df.info()\n",
        "\n",
        "print(\"\\nDescriptive Statistics:\")\n",
        "print(df.describe())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "41c5cd5e",
        "outputId": "74f92694-2fc2-4728-a149-d0f34de9037d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values before handling:\n",
            "id            0\n",
            "age           0\n",
            "sex           0\n",
            "dataset       0\n",
            "cp            0\n",
            "trestbps     59\n",
            "chol         30\n",
            "fbs          90\n",
            "restecg       2\n",
            "thalch       55\n",
            "exang        55\n",
            "oldpeak      62\n",
            "slope       309\n",
            "ca          611\n",
            "thal        486\n",
            "num           0\n",
            "dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(\"Missing values before handling:\")\n",
        "print(df.isnull().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "64c6363b",
        "outputId": "08c37806-9151-4cc1-d834-dd405ea2b8b3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dropped 'id' column. Current DataFrame columns:\n",
            "Index(['age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',\n",
            "       'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],\n",
            "      dtype='object')\n"
          ]
        }
      ],
      "source": [
        "df = df.drop('id', axis=1)\n",
        "print(\"Dropped 'id' column. Current DataFrame columns:\")\n",
        "print(df.columns)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e8efaccd",
        "outputId": "3cbfab1b-3ad9-43ba-8367-9197a7c08a69"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in 'trestbps' filled with median: 130.0\n",
            "Missing values in 'chol' filled with median: 223.0\n",
            "Missing values in 'thalch' filled with median: 140.0\n",
            "Missing values in 'oldpeak' filled with median: 0.5\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-2514112459.py:4: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  df[col].fillna(median_val, inplace=True)\n"
          ]
        }
      ],
      "source": [
        "for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:\n",
        "    if df[col].isnull().any():\n",
        "        median_val = df[col].median()\n",
        "        df[col].fillna(median_val, inplace=True)\n",
        "        print(f\"Missing values in '{col}' filled with median: {median_val}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "316c67b8"
      },
      "outputs": [],
      "source": [
        "for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:\n",
        "    if df[col].isnull().any():\n",
        "        median_val = df[col].median()\n",
        "        df[col] = df[col].fillna(median_val)\n",
        "        print(f\"Missing values in '{col}' filled with median: {median_val}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8ca247e5",
        "outputId": "b321bd0f-c0ef-4b48-f5dc-c4c12364a0c0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in 'fbs' filled with mode: False\n",
            "Missing values in 'restecg' filled with mode: normal\n",
            "Missing values in 'exang' filled with mode: False\n",
            "Missing values in 'slope' filled with mode: flat\n",
            "Missing values in 'ca' filled with mode: 0.0\n",
            "Missing values in 'thal' filled with mode: normal\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-3278397169.py:5: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
            "  df[col] = df[col].fillna(mode_val)\n"
          ]
        }
      ],
      "source": [
        "for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:\n",
        "    if df[col].isnull().any():\n",
        "        # Use .mode()[0] to get the first mode in case of multiple modes\n",
        "        mode_val = df[col].mode()[0]\n",
        "        df[col] = df[col].fillna(mode_val)\n",
        "        print(f\"Missing values in '{col}' filled with mode: {mode_val}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "f78f0f02"
      },
      "outputs": [],
      "source": [
        "for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:\n",
        "    if df[col].isnull().any():\n",
        "        # Use .mode()[0] to get the first mode in case of multiple modes\n",
        "        mode_val = df[col].mode()[0]\n",
        "        # Fill missing values and then explicitly infer object types to handle potential downcasting warnings\n",
        "        df[col] = df[col].fillna(mode_val).infer_objects(copy=False)\n",
        "        print(f\"Missing values in '{col}' filled with mode: {mode_val}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "83a2b03b",
        "outputId": "08437e14-ade3-4a42-89d9-0987bbeab411"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values after handling:\n",
            "age         0\n",
            "sex         0\n",
            "dataset     0\n",
            "cp          0\n",
            "trestbps    0\n",
            "chol        0\n",
            "fbs         0\n",
            "restecg     0\n",
            "thalch      0\n",
            "exang       0\n",
            "oldpeak     0\n",
            "slope       0\n",
            "ca          0\n",
            "thal        0\n",
            "num         0\n",
            "dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(\"Missing values after handling:\")\n",
        "print(df.isnull().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1153542f",
        "outputId": "96cd65bd-0ce9-44e7-a8ce-40699f1262df"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of duplicate rows before removal: 2\n",
            "Number of duplicate rows after removal: 0\n"
          ]
        }
      ],
      "source": [
        "print(\"Number of duplicate rows before removal:\", df.duplicated().sum())\n",
        "df.drop_duplicates(inplace=True)\n",
        "print(\"Number of duplicate rows after removal:\", df.duplicated().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9f8f62ee",
        "outputId": "9421632c-15f8-4638-a24e-24938e5df4bd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cleaned DataFrame Head:\n",
            "   age     sex    dataset               cp  trestbps   chol    fbs  \\\n",
            "0   63    Male  Cleveland   typical angina     145.0  233.0   True   \n",
            "1   67    Male  Cleveland     asymptomatic     160.0  286.0  False   \n",
            "2   67    Male  Cleveland     asymptomatic     120.0  229.0  False   \n",
            "3   37    Male  Cleveland      non-anginal     130.0  250.0  False   \n",
            "4   41  Female  Cleveland  atypical angina     130.0  204.0  False   \n",
            "\n",
            "          restecg  thalch  exang  oldpeak        slope   ca  \\\n",
            "0  lv hypertrophy   150.0  False      2.3  downsloping  0.0   \n",
            "1  lv hypertrophy   108.0   True      1.5         flat  3.0   \n",
            "2  lv hypertrophy   129.0   True      2.6         flat  2.0   \n",
            "3          normal   187.0  False      3.5  downsloping  0.0   \n",
            "4  lv hypertrophy   172.0  False      1.4    upsloping  0.0   \n",
            "\n",
            "                thal  num  \n",
            "0       fixed defect    0  \n",
            "1             normal    2  \n",
            "2  reversable defect    1  \n",
            "3             normal    0  \n",
            "4             normal    0  \n",
            "\n",
            "Cleaned DataFrame Info:\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Index: 918 entries, 0 to 919\n",
            "Data columns (total 15 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   age       918 non-null    int64  \n",
            " 1   sex       918 non-null    object \n",
            " 2   dataset   918 non-null    object \n",
            " 3   cp        918 non-null    object \n",
            " 4   trestbps  918 non-null    float64\n",
            " 5   chol      918 non-null    float64\n",
            " 6   fbs       918 non-null    bool   \n",
            " 7   restecg   918 non-null    object \n",
            " 8   thalch    918 non-null    float64\n",
            " 9   exang     918 non-null    bool   \n",
            " 10  oldpeak   918 non-null    float64\n",
            " 11  slope     918 non-null    object \n",
            " 12  ca        918 non-null    float64\n",
            " 13  thal      918 non-null    object \n",
            " 14  num       918 non-null    int64  \n",
            "dtypes: bool(2), float64(5), int64(2), object(6)\n",
            "memory usage: 102.2+ KB\n"
          ]
        }
      ],
      "source": [
        "print(\"Cleaned DataFrame Head:\")\n",
        "print(df.head())\n",
        "\n",
        "print(\"\\nCleaned DataFrame Info:\")\n",
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "73ca1da1",
        "outputId": "4cf0e5c1-72b3-4133-a9ea-3b5bd5791fad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Output directory './output' ensured.\n",
            "\n",
            "DataFrame loaded successfully. First 5 rows:\n",
            "   id  age     sex    dataset               cp  trestbps   chol    fbs  \\\n",
            "0   1   63    Male  Cleveland   typical angina     145.0  233.0   True   \n",
            "1   2   67    Male  Cleveland     asymptomatic     160.0  286.0  False   \n",
            "2   3   67    Male  Cleveland     asymptomatic     120.0  229.0  False   \n",
            "3   4   37    Male  Cleveland      non-anginal     130.0  250.0  False   \n",
            "4   5   41  Female  Cleveland  atypical angina     130.0  204.0  False   \n",
            "\n",
            "          restecg  thalch  exang  oldpeak        slope   ca  \\\n",
            "0  lv hypertrophy   150.0  False      2.3  downsloping  0.0   \n",
            "1  lv hypertrophy   108.0   True      1.5         flat  3.0   \n",
            "2  lv hypertrophy   129.0   True      2.6         flat  2.0   \n",
            "3          normal   187.0  False      3.5  downsloping  0.0   \n",
            "4  lv hypertrophy   172.0  False      1.4    upsloping  0.0   \n",
            "\n",
            "                thal  num  \n",
            "0       fixed defect    0  \n",
            "1             normal    2  \n",
            "2  reversable defect    1  \n",
            "3             normal    0  \n",
            "4             normal    0  \n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import os\n",
        "\n",
        "# Define output directory\n",
        "output_dir = './output'\n",
        "os.makedirs(output_dir, exist_ok=True)\n",
        "print(f\"Output directory '{output_dir}' ensured.\")\n",
        "\n",
        "# Load the dataset\n",
        "df = pd.read_csv('/content/heart_disease_uci.csv')\n",
        "\n",
        "# Display the first few rows to verify successful loading\n",
        "print(\"\\nDataFrame loaded successfully. First 5 rows:\")\n",
        "print(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "908be555",
        "outputId": "7456942c-9437-4d48-903e-b054b64c4e2c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DataFrame after creating 'target' and dropping specified columns:\n",
            "   age     sex               cp  trestbps   chol    fbs         restecg  \\\n",
            "0   63    Male   typical angina     145.0  233.0   True  lv hypertrophy   \n",
            "1   67    Male     asymptomatic     160.0  286.0  False  lv hypertrophy   \n",
            "2   67    Male     asymptomatic     120.0  229.0  False  lv hypertrophy   \n",
            "3   37    Male      non-anginal     130.0  250.0  False          normal   \n",
            "4   41  Female  atypical angina     130.0  204.0  False  lv hypertrophy   \n",
            "\n",
            "   thalch  exang  oldpeak        slope   ca               thal  target  \n",
            "0   150.0  False      2.3  downsloping  0.0       fixed defect       0  \n",
            "1   108.0   True      1.5         flat  3.0             normal       1  \n",
            "2   129.0   True      2.6         flat  2.0  reversable defect       1  \n",
            "3   187.0  False      3.5  downsloping  0.0             normal       0  \n",
            "4   172.0  False      1.4    upsloping  0.0             normal       0  \n"
          ]
        }
      ],
      "source": [
        "df['target'] = (df['num'] > 0).astype(int)\n",
        "df = df.drop(['id', 'dataset', 'num'], axis=1)\n",
        "print(\"DataFrame after creating 'target' and dropping specified columns:\")\n",
        "print(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1e115134",
        "outputId": "44622306-ec9d-4da5-9e89-3dbd44f92d37"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in 'trestbps' filled with median: 130.0\n",
            "Missing values in 'chol' filled with median: 223.0\n",
            "Missing values in 'thalch' filled with median: 140.0\n",
            "Missing values in 'oldpeak' filled with median: 0.5\n"
          ]
        }
      ],
      "source": [
        "for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:\n",
        "    if df[col].isnull().any():\n",
        "        median_val = df[col].median()\n",
        "        df[col] = df[col].fillna(median_val)\n",
        "        print(f\"Missing values in '{col}' filled with median: {median_val}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bce93267",
        "outputId": "75e36692-a450-49fa-d6ca-5e0aceacad88"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in 'fbs' filled with mode: False\n",
            "Missing values in 'restecg' filled with mode: normal\n",
            "Missing values in 'exang' filled with mode: False\n",
            "Missing values in 'slope' filled with mode: flat\n",
            "Missing values in 'ca' filled with mode: 0.0\n",
            "Missing values in 'thal' filled with mode: normal\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-3890388285.py:6: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
            "  df[col] = df[col].fillna(mode_val).infer_objects(copy=False)\n"
          ]
        }
      ],
      "source": [
        "for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:\n",
        "    if df[col].isnull().any():\n",
        "        # Use .mode()[0] to get the first mode in case of multiple modes\n",
        "        mode_val = df[col].mode()[0]\n",
        "        # Fill missing values and then explicitly infer object types to handle potential downcasting warnings\n",
        "        df[col] = df[col].fillna(mode_val).infer_objects(copy=False)\n",
        "        print(f\"Missing values in '{col}' filled with mode: {mode_val}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1867dceb",
        "outputId": "fd54f829-b14d-4a3c-9b6d-5d8979244a4a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values after handling:\n",
            "age         0\n",
            "sex         0\n",
            "cp          0\n",
            "trestbps    0\n",
            "chol        0\n",
            "fbs         0\n",
            "restecg     0\n",
            "thalch      0\n",
            "exang       0\n",
            "oldpeak     0\n",
            "slope       0\n",
            "ca          0\n",
            "thal        0\n",
            "target      0\n",
            "dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(\"Missing values after handling:\")\n",
        "print(df.isnull().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9380a5e5",
        "outputId": "9322e264-9e7f-4f3a-cf8d-9ee7101795a9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cleaned DataFrame saved to './output/heart_clean.csv'\n"
          ]
        }
      ],
      "source": [
        "cleaned_filepath = os.path.join(output_dir, 'heart_clean.csv')\n",
        "df.to_csv(cleaned_filepath, index=False)\n",
        "print(f\"Cleaned DataFrame saved to '{cleaned_filepath}'\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d147ad1a",
        "outputId": "0079df30-0bab-4c04-85d2-cbb103c55c4a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DataFrame after one-hot encoding categorical features:\n",
            "   age  trestbps   chol  thalch  oldpeak   ca  target  sex_Male  \\\n",
            "0   63     145.0  233.0   150.0      2.3  0.0       0      True   \n",
            "1   67     160.0  286.0   108.0      1.5  3.0       1      True   \n",
            "2   67     120.0  229.0   129.0      2.6  2.0       1      True   \n",
            "3   37     130.0  250.0   187.0      3.5  0.0       0      True   \n",
            "4   41     130.0  204.0   172.0      1.4  0.0       0     False   \n",
            "\n",
            "   cp_atypical angina  cp_non-anginal  cp_typical angina  fbs_True  \\\n",
            "0               False           False               True      True   \n",
            "1               False           False              False     False   \n",
            "2               False           False              False     False   \n",
            "3               False            True              False     False   \n",
            "4                True           False              False     False   \n",
            "\n",
            "   restecg_normal  restecg_st-t abnormality  exang_True  slope_flat  \\\n",
            "0           False                     False       False       False   \n",
            "1           False                     False        True        True   \n",
            "2           False                     False        True        True   \n",
            "3            True                     False       False       False   \n",
            "4           False                     False       False       False   \n",
            "\n",
            "   slope_upsloping  thal_normal  thal_reversable defect  \n",
            "0            False        False                   False  \n",
            "1            False         True                   False  \n",
            "2            False        False                    True  \n",
            "3            False         True                   False  \n",
            "4             True         True                   False  \n"
          ]
        }
      ],
      "source": [
        "categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']\n",
        "df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
        "print(\"DataFrame after one-hot encoding categorical features:\")\n",
        "print(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "12df5b7a",
        "outputId": "30320e15-99ae-4fe0-ffa1-827f1f7b2802"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DataFrame after scaling numerical features:\n",
            "        age  trestbps      chol    thalch   oldpeak        ca  target  \\\n",
            "0  1.007386  0.705176  0.303643  0.489727  1.368109 -0.361400       0   \n",
            "1  1.432034  1.518569  0.789967 -1.181478  0.611589  4.411152       1   \n",
            "2  1.432034 -0.650479  0.266939 -0.345875  1.651804  2.820301       1   \n",
            "3 -1.752828 -0.108217  0.459634  1.961979  2.502889 -0.361400       0   \n",
            "4 -1.328180 -0.108217  0.037541  1.365120  0.517024 -0.361400       0   \n",
            "\n",
            "   sex_Male  cp_atypical angina  cp_non-anginal  cp_typical angina  fbs_True  \\\n",
            "0      True               False           False               True      True   \n",
            "1      True               False           False              False     False   \n",
            "2      True               False           False              False     False   \n",
            "3      True               False            True              False     False   \n",
            "4     False                True           False              False     False   \n",
            "\n",
            "   restecg_normal  restecg_st-t abnormality  exang_True  slope_flat  \\\n",
            "0           False                     False       False       False   \n",
            "1           False                     False        True        True   \n",
            "2           False                     False        True        True   \n",
            "3            True                     False       False       False   \n",
            "4           False                     False       False       False   \n",
            "\n",
            "   slope_upsloping  thal_normal  thal_reversable defect  \n",
            "0            False        False                   False  \n",
            "1            False         True                   False  \n",
            "2            False        False                    True  \n",
            "3            False         True                   False  \n",
            "4             True         True                   False  \n"
          ]
        }
      ],
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "# Identify numerical columns to scale (excluding 'target')\n",
        "numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']\n",
        "\n",
        "# Initialize StandardScaler\n",
        "scaler = StandardScaler()\n",
        "\n",
        "# Apply scaling to numerical columns\n",
        "df[numerical_cols] = scaler.fit_transform(df[numerical_cols])\n",
        "\n",
        "print(\"DataFrame after scaling numerical features:\")\n",
        "print(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2267df4a",
        "outputId": "b34443f3-8ee4-41f6-a22e-06955588f89e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Data split into training and testing sets.\n",
            "X_train shape: (736, 18), y_train shape: (736,)\n",
            "X_test shape: (184, 18), y_test shape: (184,)\n",
            "RandomForestClassifier model trained successfully.\n"
          ]
        }
      ],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# Separate features (X) and target (y)\n",
        "X = df.drop('target', axis=1)\n",
        "y = df['target']\n",
        "\n",
        "# Split data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "print(\"Data split into training and testing sets.\")\n",
        "print(f\"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\")\n",
        "print(f\"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\")\n",
        "\n",
        "# Initialize and train the RandomForestClassifier model\n",
        "model = RandomForestClassifier(random_state=42)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "print(\"RandomForestClassifier model trained successfully.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d19560fd",
        "outputId": "b1714b7b-a7f7-4576-d605-8ec02d2a665d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model Evaluation:\n",
            "\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.88      0.79      0.83        82\n",
            "           1       0.85      0.91      0.88       102\n",
            "\n",
            "    accuracy                           0.86       184\n",
            "   macro avg       0.86      0.85      0.86       184\n",
            "weighted avg       0.86      0.86      0.86       184\n",
            "\n",
            "\n",
            "ROC-AUC Score: 0.9198\n",
            "\n",
            "Confusion Matrix:\n",
            "[[65 17]\n",
            " [ 9 93]]\n"
          ]
        }
      ],
      "source": [
        "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n",
        "\n",
        "# Make predictions on the test data\n",
        "y_pred = model.predict(X_test)\n",
        "y_proba = model.predict_proba(X_test)[:, 1]\n",
        "\n",
        "print(\"Model Evaluation:\")\n",
        "\n",
        "# Print Classification Report\n",
        "print(\"\\nClassification Report:\")\n",
        "print(classification_report(y_test, y_pred))\n",
        "\n",
        "# Calculate and print ROC-AUC Score\n",
        "roc_auc = roc_auc_score(y_test, y_proba)\n",
        "print(f\"\\nROC-AUC Score: {roc_auc:.4f}\")\n",
        "\n",
        "# Calculate and print Confusion Matrix\n",
        "conf_matrix = confusion_matrix(y_test, y_pred)\n",
        "print(\"\\nConfusion Matrix:\")\n",
        "print(conf_matrix)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "HRtasniOSfo2"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "c5f0ef43",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ab6c2f23-ee49-4d91-a63d-cebaeb1eff6d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: line 1: streamlit: command not found\n"
          ]
        }
      ],
      "source": [
        "!streamlit run app.py"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "20bd74cb",
        "outputId": "9e7e5cac-d166-4acc-b981-4ec499d86d4d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: line 1: streamlit: command not found\n"
          ]
        }
      ],
      "source": [
        "!streamlit run app.py"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "a5983a18"
      },
      "outputs": [],
      "source": [
        "# To run the Streamlit app, execute the following in your terminal:\n",
        "#  !streamlit run app.py\n",
        "# This will provide a local URL and a network URL. Use the network URL to access the app in your browser."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f4687372",
        "outputId": "898512f7-1935-48b3-b87f-693fc4e95e12"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import joblib # Often used for saving/loading models/scalers\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "import os\n",
        "\n",
        "# --- Streamlit App Setup ---\n",
        "st.title(\"Heart Disease Prediction\")\n",
        "st.write(\"Enter patient information to predict the likelihood of heart disease.\")\n",
        "\n",
        "# --- Simulate loading of pre-trained model and scaler ---\n",
        "# In a real application, these objects (model, scaler) and the `X_train_columns` list\n",
        "# would be saved to files (e.g., using joblib) after training and then loaded here.\n",
        "# For the purpose of this exercise, we will re-create them based on the previous notebook steps\n",
        "# to ensure the app is self-contained and runnable.\n",
        "\n",
        "# 1. Load original data\n",
        "# Assume '/content/heart_disease_uci.csv' is available as per the environment.\n",
        "original_df = pd.read_csv('/content/heart_disease_uci.csv')\n",
        "\n",
        "# 2. Prepare target and drop columns ('id', 'dataset', 'num')\n",
        "df_temp_for_processing = original_df.copy()\n",
        "df_temp_for_processing['target'] = (df_temp_for_processing['num'] > 0).astype(int)\n",
        "df_temp_for_processing = df_temp_for_processing.drop(['id', 'dataset', 'num'], axis=1)\n",
        "\n",
        "# 3. Handle missing values (Numerical with median, Categorical with mode)\n",
        "# Define columns based on prior analysis\n",
        "numerical_cols_for_processing = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']\n",
        "categorical_cols_for_processing = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']\n",
        "\n",
        "for col in numerical_cols_for_processing:\n",
        "    if df_temp_for_processing[col].isnull().any():\n",
        "        median_val = df_temp_for_processing[col].median()\n",
        "        df_temp_for_processing[col] = df_temp_for_processing[col].fillna(median_val)\n",
        "\n",
        "for col in categorical_cols_for_processing:\n",
        "    if df_temp_for_processing[col].isnull().any():\n",
        "        mode_val = df_temp_for_processing[col].mode()[0]\n",
        "        df_temp_for_processing[col] = df_temp_for_processing[col].fillna(mode_val).infer_objects(copy=False)\n",
        "\n",
        "# 4. Fit StandardScaler on numerical features *before* one-hot encoding for the entire dataset\n",
        "scaler = StandardScaler()\n",
        "scaler.fit(df_temp_for_processing[numerical_cols_for_processing])\n",
        "\n",
        "# 5. Apply one-hot encoding to categorical features for the full processed dataframe\n",
        "df_processed_final = pd.get_dummies(df_temp_for_processing, columns=categorical_cols_for_processing, drop_first=True)\n",
        "\n",
        "# 6. Separate features (X) and target (y) for model training\n",
        "X_full_for_training = df_processed_final.drop('target', axis=1)\n",
        "y_full_for_training = df_processed_final['target']\n",
        "\n",
        "# Store the column names for alignment later\n",
        "X_train_columns = X_full_for_training.columns.tolist()\n",
        "\n",
        "# 7. Train the RandomForestClassifier model\n",
        "# In a real app, this would be `model = joblib.load('random_forest_model.pkl')`\n",
        "model = RandomForestClassifier(random_state=42)\n",
        "model.fit(X_full_for_training, y_full_for_training)\n",
        "\n",
        "# --- End of Simulation for Loaded Objects ---\n",
        "\n",
        "# Define the numerical and categorical columns that were used in feature engineering\n",
        "# These lists are used for processing user input consistently with training data\n",
        "numerical_features_app = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']\n",
        "categorical_features_app = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']\n",
        "\n",
        "# --- Streamlit Input Fields ---\n",
        "with st.sidebar:\n",
        "    st.header(\"Patient Data Input\")\n",
        "\n",
        "    age = st.slider(\"Age\", 20, 80, 50)\n",
        "    sex_input = st.radio(\"Sex\", ['Male', 'Female'])\n",
        "    cp_input = st.selectbox(\"Chest Pain Type (cp)\", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])\n",
        "    trestbps = st.slider(\"Resting Blood Pressure (trestbps)\", 90, 200, 120)\n",
        "    chol = st.slider(\"Serum Cholestoral (chol)\", 100, 600, 200)\n",
        "    fbs_input = st.checkbox(\"Fasting Blood Sugar > 120 mg/dl (fbs)\", False)\n",
        "    restecg_input = st.selectbox(\"Resting Electrocardiographic Results (restecg)\", ['normal', 'st-t abnormality', 'lv hypertrophy'])\n",
        "    thalch = st.slider(\"Maximum Heart Rate Achieved (thalch)\", 70, 210, 150)\n",
        "    exang_input = st.checkbox(\"Exercise Induced Angina (exang)\", False)\n",
        "    oldpeak = st.slider(\"ST depression induced by exercise relative to rest (oldpeak)\", 0.0, 6.0, 1.0, 0.1)\n",
        "    slope_input = st.selectbox(\"The slope of the peak exercise ST segment (slope)\", ['upsloping', 'flat', 'downsloping'])\n",
        "    ca = st.slider(\"Number of major vessels (0-3) colored by flourosopy (ca)\", 0, 3, 0)\n",
        "    thal_input = st.selectbox(\"Thal\", ['normal', 'fixed defect', 'reversable defect'])\n",
        "\n",
        "    predict_button = st.button(\"Predict Heart Disease\")\n",
        "\n",
        "# --- Prediction Logic ---\n",
        "if predict_button:\n",
        "    # 1. Create a Pandas DataFrame from the user's input values.\n",
        "    user_input_df = pd.DataFrame({\n",
        "        'age': [age],\n",
        "        'sex': [sex_input],\n",
        "        'cp': [cp_input],\n",
        "        'trestbps': [float(trestbps)], # Ensure type consistency\n",
        "        'chol': [float(chol)],         # Ensure type consistency\n",
        "        'fbs': [fbs_input],\n",
        "        'restecg': [restecg_input],\n",
        "        'thalch': [float(thalch)],     # Ensure type consistency\n",
        "        'exang': [exang_input],\n",
        "        'oldpeak': [float(oldpeak)],   # Ensure type consistency\n",
        "        'slope': [slope_input],\n",
        "        'ca': [float(ca)],             # Ensure type consistency\n",
        "        'thal': [thal_input]\n",
        "    })\n",
        "\n",
        "    st.write(\"### User Input Data:\")\n",
        "    st.dataframe(user_input_df)\n",
        "\n",
        "    # 2. Apply one-hot encoding to the categorical features in the user input DataFrame.\n",
        "    user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_features_app, drop_first=True)\n",
        "\n",
        "    # 3. Align the columns of the one-hot encoded user input DataFrame with the columns of X_train_columns.\n",
        "    # This ensures that the user input DataFrame has the same columns in the same order as the training data.\n",
        "    user_input_aligned = user_input_encoded.reindex(columns=X_train_columns, fill_value=0)\n",
        "\n",
        "    # 4. & 5. Scale the numerical features in the aligned user input DataFrame.\n",
        "    user_input_aligned[numerical_features_app] = scaler.transform(user_input_aligned[numerical_features_app])\n",
        "\n",
        "    st.write(\"### Preprocessed User Input for Prediction:\")\n",
        "    st.dataframe(user_input_aligned)\n",
        "\n",
        "    # 6. Use the loaded `model` to make a prediction.\n",
        "    prediction = model.predict(user_input_aligned)\n",
        "    prediction_proba = model.predict_proba(user_input_aligned)[:, 1] # Probability of heart disease (class 1)\n",
        "\n",
        "    # 7. Display the prediction result.\n",
        "    st.write(\"---\")\n",
        "    st.subheader(\"Prediction Result:\")\n",
        "    if prediction[0] == 1:\n",
        "        st.error(f\"The model predicts **Heart Disease** with a probability of {prediction_proba[0]:.2f}\")\n",
        "    else:\n",
        "        st.success(f\"The model predicts **No Heart Disease** with a probability of {1 - prediction_proba[0]:.2f}\")\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
