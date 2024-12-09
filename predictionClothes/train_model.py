import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and annotate the data
data = pd.read_csv('data.csv')

def generate_outfit_labels(row):
    # Shoes
    if row['rain_sum'] > 5:
        shoes = "waterproof boots"
    elif row['rain_sum'] > 0:
        shoes = "boots"
    elif row['temp_min'] < 5:
        shoes = "warm boots"
    else:
        shoes = "sneakers"

    # Lower body clothes
    if row['temp_max'] > 25:
        lower_body = "shorts"
    elif row['temp_max'] > 15:
        lower_body = "light pants"
    else:
        lower_body = "jeans"

    # Upper body clothes
    if row['temp_max'] > 25:
        upper_body = "t-shirt"
    elif row['temp_max'] > 15:
        upper_body = "light jacket"
    else:
        upper_body = "heavy jacket"

    return shoes, lower_body, upper_body

data['shoes'], data['lower_body'], data['upper_body'] = zip(*data.apply(generate_outfit_labels, axis=1))

# Feature extraction and label encoding
features = ['temp_max', 'temp_min', 'uv_index_max', 'rain_sum']
X = data[features]

label_encoders = {col: LabelEncoder() for col in ['shoes', 'lower_body', 'upper_body']}
y_shoes = label_encoders['shoes'].fit_transform(data['shoes'])
y_lower_body = label_encoders['lower_body'].fit_transform(data['lower_body'])
y_upper_body = label_encoders['upper_body'].fit_transform(data['upper_body'])

# Train-test split
X_train, X_test, y_shoes_train, y_shoes_test = train_test_split(X, y_shoes, test_size=0.2, random_state=42)
_, _, y_lower_body_train, y_lower_body_test = train_test_split(X, y_lower_body, test_size=0.2, random_state=42)
_, _, y_upper_body_train, y_upper_body_test = train_test_split(X, y_upper_body, test_size=0.2, random_state=42)

# Train models
shoe_model = RandomForestClassifier(random_state=42)
lower_body_model = RandomForestClassifier(random_state=42)
upper_body_model = RandomForestClassifier(random_state=42)

shoe_model.fit(X_train, y_shoes_train)
lower_body_model.fit(X_train, y_lower_body_train)
upper_body_model.fit(X_train, y_upper_body_train)

# Save models and encoders
import joblib
joblib.dump(shoe_model, 'shoe_model.pkl')
joblib.dump(lower_body_model, 'lower_body_model.pkl')
joblib.dump(upper_body_model, 'upper_body_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
