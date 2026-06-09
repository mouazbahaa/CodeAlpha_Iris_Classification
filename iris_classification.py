# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# قراءة ملف البيانات المستخرج باستخدام المسار المباشر
data_path = r"D:\internships\codealpha\task 1\Iris.csv"
df = pd.read_csv(data_path)

# حذف عمود الـ Id لأنه مش مفيد للموديل
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# طباعة أول 5 صفوف في الـ Console
print("--- أول 5 صفوف في الداتا ست ---")
print(df.head())

# طباعة معلومات عامة عن البيانات
print("\n--- معلومات عامة عن البيانات ---")
print(df.info())
import matplotlib.pyplot as plt

# 5. رسم بياني يوضح توزيع الفصائل بناءً على مقاسات الـ Petal
print("\n--- جاري تجهيز الرسم البياني باستخدام Matplotlib... ---")

# تقسيم البيانات حسب الفصيلة عشان نلون كل واحدة بلون مختلف
setosa = df[df['Species'] == 'Iris-setosa']
versicolor = df[df['Species'] == 'Iris-versicolor']
virginica = df[df['Species'] == 'Iris-virginica']

# إنشاء الرسمة
plt.figure(figsize=(8, 6))
plt.scatter(setosa['PetalLengthCm'], setosa['PetalWidthCm'], color='red', label='Setosa')
plt.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], color='blue', label='Versicolor')
plt.scatter(virginica['PetalLengthCm'], virginica['PetalWidthCm'], color='green', label='Virginica')

# إضافات لتوضيح الرسمة
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Flower Species Distribution')
plt.legend()
plt.grid(True)

# عرض الرسمة في الـ Plots جوه Spyder
plt.show()
# 6. تقسيم البيانات يدوياً إلى 80% تدريب و 20% اختبار لضمان التقييم الصح
# هنعمل Shuffle (ترتيب عشوائي) للداتا أولاً
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(df_shuffled) * 0.8)
train_set = df_shuffled.iloc[:train_size]
test_set = df_shuffled.iloc[train_size:]

# 7. بناء موديل شجرة القرار يدوياً (Rule-based Classifier) بناءً على توزيع الرسم البياني
def predict_flower(petal_length, petal_width):
    # لو طول البتلة أصغر من 2.5 سم إذن هي بالتأكيد Setosa (اللون الأحمر في الرسمة)
    if petal_length < 2.5:
        return 'Iris-setosa'
    # لو عرض البتلة أكبر من 1.75 سم إذن هي في الغالب Virginica (اللون الأخضر)
    elif petal_width > 1.75:
        return 'Iris-virginica'
    # غير كده تبقى الفصيلة المتوسطة Versicolor (اللون الأزرق)
    else:
        return 'Iris-versicolor'

# 8. اختبار الموديل على بيانات الـ Test المستقلة
correct_predictions = 0

for idx, row in test_set.iterrows():
    prediction = predict_flower(row['PetalLengthCm'], row['PetalWidthCm'])
    if prediction == row['Species']:
        correct_predictions += 1

# 9. حساب الدقة (Accuracy) وطباعتها
total_test = len(test_set)
accuracy = (correct_predictions / total_test) * 100

print("\n--- تقييم الموديل الذكي (From Scratch) ---")
print(f"Total Test Samples: {total_test}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Model Accuracy: {accuracy:.2f}%")
