import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("⏳ Loading data...")
# 1. قراءة البيانات
df = pd.read_csv('employee_data.csv')

# 2. اختيار أهم الأعمدة (Features) لتسهيل الواجهة
# Age, MonthlyIncome, JobSatisfaction, YearsAtCompany, OverTime, DistanceFromHome, WorkLifeBalance
features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 
            'OverTime', 'DistanceFromHome', 'WorkLifeBalance']

X = df[features].copy()
y = df['Attrition']

print("⚙️ Preprocessing data...")
# تحويل النصوص إلى أرقام يفهمها النموذج
# Attrition: Yes = 1, No = 0
y = y.map({'Yes': 1, 'No': 0})

# OverTime: Yes = 1, No = 0
X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

# 3. تقسيم البيانات إلى قسم تدريب (80%) وقسم اختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest Model...")
# 4. تدريب خوارزمية Random Forest
# استخدمنا class_weight='balanced' لأن الموظفين اللي بيستقيلوا أقل من اللي بيبقوا
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. اختبار دقة النموذج
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# 6. حفظ النموذج في ملف لكي يستخدمه تطبيق Streamlit لاحقاً
with open('attrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved as 'attrition_model.pkl'. You can now use it in your app!")