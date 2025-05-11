import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif agar bisa dijalankan di server/headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import argparse

def preprocess_data(file_path, target_column, save_path=None):
    # Load dataset dari file CSV
    data = pd.read_csv(file_path)

    # Tangani missing value pada kolom "parental_education_level" jika ada
    if 'parental_education_level' in data.columns:
        data['parental_education_level'] = data['parental_education_level'].fillna(
            data['parental_education_level'].mode()[0]
        )

    # Hapus kolom "student_id" jika ada, karena tidak relevan untuk analisis
    if 'student_id' in data.columns:
        data.drop('student_id', axis=1, inplace=True)

    # Identifikasi kolom numerik dan kategorikal
    num_col = data.select_dtypes(include='number').columns
    cat_col = data.select_dtypes(include='object').columns

    # Buat direktori untuk menyimpan file jika belum ada
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Simpan visualisasi distribusi data numerik
    plt.figure(figsize=(12, 12))
    for i in range(min(len(num_col), 9)):
        plt.subplot(3, 3, i + 1)
        plt.hist(data[num_col[i]], bins=20, edgecolor='black')
        plt.title(f'Distribusi dari "{num_col[i]}"')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "numeric_distribution.png"))
    plt.close()

    # Simpan visualisasi distribusi data kategorikal
    plt.figure(figsize=(10, 6))
    for i in range(min(len(cat_col), 6)):
        plt.subplot(2, 3, i + 1)
        plt.hist(data[cat_col[i]], color='skyblue', edgecolor='black')
        plt.title(f"Distribusi dari '{cat_col[i]}'")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "categorical_distribution.png"))
    plt.close()

    # Salin dataset agar tidak mengubah data asli
    data2 = data.copy()

    # Mapping manual untuk fitur kategorikal ordinal menjadi numerik
    diet_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    parental_education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}
    internet_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}

    data2['dq_e'] = data2['diet_quality'].map(diet_quality_map)
    data2['pel_e'] = data2['parental_education_level'].map(parental_education_map)
    data2['iq_e'] = data2['internet_quality'].map(internet_quality_map)

    # One-hot encoding untuk fitur nominal dan hindari dummy trap
    dummies = pd.get_dummies(data2[['gender', 'part_time_job', 'extracurricular_participation']], drop_first=True)

    # Gabungkan one-hot encoded columns ke dataset utama
    data3 = pd.concat([data2, dummies], axis=1)

    # Hapus kolom aslinya karena sudah direpresentasikan dalam bentuk numerik/encoded
    data3.drop(['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                'internet_quality', 'extracurricular_participation'], axis=1, inplace=True)

    # Pastikan kolom target ada
    if target_column not in data3.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam data.")

    # Simpan visualisasi heatmap korelasi
    plt.figure(figsize=(12, 10))
    sns.heatmap(data3.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='gray')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Heatmap Korelasi antar Variabel", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "correlation_heatmap.png"))
    plt.close()

    # Pisahkan fitur dan target
    X = data3.drop(target_column, axis=1)
    y = data3[target_column]

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simpan preprocessor dan gabungan hasil X + y
    if save_path:
        dump(scaler, os.path.join(save_path, "preprocessor.joblib"))
        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df[target_column] = y.values
        processed_df.to_csv(os.path.join(save_path, "processed_data.csv"), index=False)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess student habit dataset")
    parser.add_argument("--file_path", type=str, required=True, help="Path ke file dataset .csv")
    parser.add_argument("--target_column", type=str, default="exam_score", help="Kolom target yang akan diprediksi")
    parser.add_argument("--save_path", type=str, required=True, help="Direktori untuk menyimpan hasil preprocessing")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = preprocess_data(
        file_path=args.file_path,
        target_column=args.target_column,
        save_path=args.save_path
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)