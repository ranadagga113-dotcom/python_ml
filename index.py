import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# [0] قراءة الملف مباشرة من سطح المكتب
df = pd.read_csv(r'C:\Users\hp\Desktop\market_data.csv')

# [1] التحقق من البيانات الأساسية
print("أبعاد جدول البيانات:", df.shape)

print("---طباعة أول 5 صفوف---")
print(df.head(5))

print("---طباعة اخر 5 صفوف---")
print(df.tail(5))

print("---طباعة معلومات الجدول---")
print(df.info())

print("---طباعة الاحصائيات الوصفية---")
print(df.describe())

stats_per_item = df.groupby('item')['price_shekel'].agg(['mean', 'median', 'std', 'min', 'max'])
print("---طباعة احصائيات لكل صنف---")
print(stats_per_item)

mean_by_border = df.groupby('border_open')['price_shekel'].mean()
print('---متوسط السعر حسب حالة المعبر---')
print(mean_by_border)

mean_by_crisis = df.groupby('crisis_level')['price_shekel'].mean()
print('---متوسط السعر حسب مستوى الازمة---')
print(mean_by_crisis)

pivot_border_crisis = df.pivot_table(index='crisis_level', columns='border_open', values='price_shekel', aggfunc='mean')
print("---جدول تقاطع(المعبرxالازمة)---") 
print(pivot_border_crisis) 

availability_pct = df['availability'].value_counts(normalize=True) * 100
print("---توزيع حالات التوفر---")
print(availability_pct)

pivot_prices = df.pivot_table(index='date', columns='item', values='price_shekel')
print('---انشاء مصفوفة محورية---')
print(pivot_prices.head())

normalized_prices = pivot_prices.div(pivot_prices.median())
print("---تطبيع الاسعار---")
print(normalized_prices.head())

price_index = normalized_prices.mean(axis=1)

scaler = StandardScaler()
price_index_scaled = scaler.fit_transform(price_index.values.reshape(-1, 1))

# [2] Clustering & Elbow Method
inertia = []
k_range = range(2, 5)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(price_index_scaled)
    inertia.append(kmeans_test.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('عدد المجموعات (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method Test')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(price_index_scaled)

temp_df = pd.DataFrame({'date': price_index.index, 'cluster': clusters, 'price_index': price_index.values})
cluster_means = temp_df.groupby('cluster')['price_index'].mean().sort_values()

cluster_mapping = {
    cluster_means.index[0]: 'رخاء',
    cluster_means.index[1]: 'غلاء متوسط',
    cluster_means.index[2]: 'أزمة'
}

temp_df['state'] = temp_df['cluster'].map(cluster_mapping)
df = df.merge(temp_df[['date', 'state']], on='date', how='left')

sil_score = silhouette_score(price_index_scaled, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Inertia: {kmeans.inertia_:.4f}")

summary_table = temp_df.groupby('state')['price_index'].agg(['count', 'mean'])
print("\n--- جدول ملخص حالات السوق ---")
print(summary_table)

merge_analysis = df.groupby('state')[['border_open', 'crisis_level']].mean()
print("--- تحليل الربط مع المعابر والأزمات ---")
print(merge_analysis)

# [3] Anomaly Detection
availability_mapping = {
    'متوفر': 2,
    'ناقص': 1,
    'شحيح': 0
}
df['avail_num'] = df['availability'].map(availability_mapping)

features = ['price_shekel', 'usd_rate', 'border_open', 'crisis_level', 'avail_num']
X_anomaly = df[features]

scaler_anomaly = StandardScaler()
X_scaled = scaler_anomaly.fit_transform(X_anomaly)

euclidean_distances = np.sqrt(np.sum(X_scaled**2, axis=1))
df['anomaly_score'] = euclidean_distances

threshold = np.percentile(euclidean_distances, 95)
df['anomaly'] = np.where(df['anomaly_score'] > threshold, -1, 1)

anomalies_count = (df['anomaly'] == -1).sum()
total_rows = len(df)
anomalies_percentage = (anomalies_count / total_rows) * 100

print(f"عدد الصفوف الشاذة المكتشفة: {anomalies_count}")
print(f"نسبة الشواذ من إجمالي البيانات: {anomalies_percentage:.2f}%")

print("\n--- أول 10 صفوف شاذة مكتشفة ---")
anomalies_df = df[df['anomaly'] == -1]
display(anomalies_df.head(10))

plt.figure(figsize=(10, 6))
normal_points = df[df['anomaly'] == 1]
plt.scatter(normal_points['usd_rate'], normal_points['price_shekel'], color='blue', alpha=0.5, label='طبيعي (Normal)')

anomaly_points = df[df['anomaly'] == -1]
plt.scatter(anomaly_points['usd_rate'], anomaly_points['price_shekel'], color='red', marker='x', s=100, label='شاذ (Anomaly)')

plt.xlabel('سعر صرف الدولار (usd_rate)')
plt.ylabel('سعر السلعة بالشيكل (price_shekel)')
plt.title('مخطط تشتت كشف الشذوذ (Anomaly Detection)')
plt.legend()
plt.grid(True)
plt.show()

# [4] Regression Model
df = df.sort_values(by=['item', 'date']).reset_index(drop=True)
df['price_next'] = df.groupby('item')['price_shekel'].shift(-1)
df_reg = df.dropna(subset=['price_next']).copy()
features_reg = ['price_shekel', 'usd_rate', 'border_open', 'crisis_level', 'avail_num']
X_reg = df_reg[features_reg]
y_reg = df_reg['price_next']

split_index = int(len(X_reg) * 0.8)
X_train_reg, X_test_reg = X_reg.iloc[:split_index], X_reg.iloc[split_index:]
y_train_reg, y_test_reg = y_reg.iloc[:split_index], y_reg.iloc[split_index:]

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg_scaled)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)
mape = np.mean(np.abs((y_test_reg - y_pred_reg) / y_test_reg)) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

plt.figure(figsize=(8, 4))
pd.Series(lin_reg.coef_, index=features_reg).plot(kind='bar')
plt.title('Linear Regression Coefficients')
plt.show()

# [5] Classification Model
features_clf = ['price_shekel', 'usd_rate', 'border_open', 'crisis_level']
X_clf = df[features_clf]
y_clf = df['availability']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf = log_reg.predict(X_test_clf_scaled)

print(classification_report(y_test_clf, y_pred_clf))

cm = confusion_matrix(y_test_clf, y_pred_clf, labels=['متوفر', 'ناقص', 'شحيح'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['متوفر', 'ناقص', 'شحيح'], 
            yticklabels=['متوفر', 'ناقص', 'شحيح'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# [6] Prediction Functions & Scenarios
def predict_price_next(price_today, usd, border, crisis, avail):
    avail_mapping = {'متوفر': 2, 'ناقص': 1, 'شحيح': 0}
    avail_num_val = avail_mapping.get(avail, 2)
    input_data = np.array([[price_today, usd, border, crisis, avail_num_val]])
    input_scaled = scaler_reg.transform(input_data)
    return lin_reg.predict(input_scaled)[0]

def predict_availability(price_today, usd, border, crisis):
    input_data = np.array([[price_today, usd, border, crisis]])
    input_scaled = scaler_clf.transform(input_data)
    return log_reg.predict(input_scaled)[0]

scenarios = [
    {"desc": "استقرار ورخاء", "price": 120, "usd": 3.70, "border": 1, "crisis": 0, "avail": "متوفر"},
    {"desc": "تصعيد متوسط", "price": 150, "usd": 3.85, "border": 1, "crisis": 1, "avail": "ناقص"},
    {"desc": "أزمة واغلاق كامل", "price": 180, "usd": 3.95, "border": 0, "crisis": 2, "avail": "شحيح"}
]

for s in scenarios:
    p_next = predict_price_next(s["price"], s["usd"], s["border"], s["crisis"], s["avail"])
    a_pred = predict_availability(s["price"], s["usd"], s["border"], s["crisis"])
    print(f"السيناريو: {s['desc']} | السعر المتوقع لغد: {p_next:.2f} | التوفر المتوقع: {a_pred}")
