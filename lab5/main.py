import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def normalize_data(train_data, test_data):
    scaler = StandardScaler()
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    return normalized_train_data, normalized_test_data


def atr_name(idx):
    if idx == 0: return "Year"
    if idx == 1: return "Kilometers_Driven"
    if idx == 2: return "Mileage"
    if idx == 3: return "Engine"
    if idx == 4: return "Power"
    if idx == 5: return "Seats"
    if idx == 6: return "Owner_Type"
    if idx >= 7 and idx <= 11: return f"Fuel_Type_OneHot_{idx-6}"
    return f"Transmission_Type_OneHot_{idx-11}"


def lin_reg(training_data, prices):
    kf = KFold(n_splits=3)
    mse_scores = []
    mae_scores = []

    model = LinearRegression()

    for train_idx, val_idx in kf.split(training_data):
        X_train, X_val = training_data[train_idx], training_data[val_idx]
        y_train, y_val = prices[train_idx], prices[val_idx]

        X_train_normalized, X_val_normalized = normalize_data(X_train, X_val)

        model.fit(X_train_normalized, y_train)
        predictions = model.predict(X_val_normalized)

        mse_scores.append(mean_squared_error(y_val, predictions))
        mae_scores.append(mean_absolute_error(y_val, predictions))

    print(f"Mean MSE for Linear Regression: {np.mean(mse_scores):.6f}")
    print(f"Mean MAE for Linear Regression: {np.mean(mae_scores):.6f}\n")


def ridge_reg(training_data, prices, alpha):
    model = Ridge(alpha=alpha)
    mse_scores = []
    mae_scores = []

    kf = KFold(n_splits=3)
    for train_idx, val_idx in kf.split(training_data):
        X_train, X_val = training_data[train_idx], training_data[val_idx]
        y_train, y_val = prices[train_idx], prices[val_idx]

        X_train_normalized, X_val_normalized = normalize_data(X_train, X_val)

        model.fit(X_train_normalized, y_train)
        predictions = model.predict(X_val_normalized)

        mse_scores.append(mean_squared_error(y_val, predictions))
        mae_scores.append(mean_absolute_error(y_val, predictions))

    return mse_scores, mae_scores


def main():
    training_data = np.load('data/training_data.npy')
    prices = np.load('data/prices.npy')
    training_data, prices = shuffle(training_data, prices, random_state=0)

    lin_reg(training_data, prices)

    alphas = [1, 10, 100, 1000]
    best_alpha = None
    best_mean_mse_ridge = float('inf')
    best_mean_mae_ridge = float('inf')

    for alpha in alphas:
        mse_scores, mae_scores = ridge_reg(training_data, prices, alpha)

        mean_mse_ridge = np.mean(mse_scores)
        mean_mae_ridge = np.mean(mae_scores)

        print(f"Alpha: {alpha}")
        print(f"  Mean MSE for Ridge Regression: {mean_mse_ridge:.6f}")
        print(f"  Mean MAE for Ridge Regression: {mean_mae_ridge:.6f}")

        if mean_mse_ridge < best_mean_mse_ridge:
            best_mean_mse_ridge = mean_mse_ridge
            best_mean_mae_ridge = mean_mae_ridge
            best_alpha = alpha

    print(f"\nBest Alpha for Ridge Regression: {best_alpha}")
    print(f"Best Mean MSE for Ridge Regression: {best_mean_mse_ridge:.6f}")
    print(f"Best Mean MAE for Ridge Regression: {best_mean_mae_ridge:.6f}\n")

    scaler = StandardScaler()
    training_data_normalized = scaler.fit_transform(training_data)

    final_ridge_model = Ridge(alpha=best_alpha)
    final_ridge_model.fit(training_data_normalized, prices)
    print(f"Coefs: {final_ridge_model.coef_}")
    print(f"\nBias: {final_ridge_model.intercept_:.6f}")

    abs_coefficients = np.abs(final_ridge_model.coef_)
    sorted_indices = np.argsort(abs_coefficients)[::-1]

    print("\nAtributes:")

    most_significant_idx = sorted_indices[0]
    most_significant_atr = atr_name(most_significant_idx)
    print(f"  Most significant attribute: {most_significant_atr} (Coef: {final_ridge_model.coef_[most_significant_idx]:.6f})")

    second_significant_idx = sorted_indices[1]
    second_significant_atr = atr_name(second_significant_idx)
    print(f"  Second significant attribute: {second_significant_atr} (Coef: {final_ridge_model.coef_[second_significant_idx]:.6f})")

    least_significant_idx = sorted_indices[-1]
    least_significant_atr = atr_name(least_significant_idx)
    print(f"  Least significant attribute: {least_significant_atr} (Coef: {final_ridge_model.coef_[least_significant_idx]:.6f})")

if __name__ == "__main__":
    main()
