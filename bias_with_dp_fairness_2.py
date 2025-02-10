import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
from typing import List
import matplotlib.pyplot as plt

# ----- DIFF PRIVACY -----
try:
    from opacus import PrivacyEngine
except ImportError:
    from opacus.privacy_engine import PrivacyEngine

# ----- AIF360 FOR FAIRNESS -----
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_run_seeds(base_seed: int, num_runs: int) -> List[int]:
    random.seed(base_seed)
    np.random.seed(base_seed)
    seeds = random.sample(range(1, 1000000), num_runs)
    return seeds


# ------------------
# Dataset + Preprocessing
# ------------------

class AdultDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_loan_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, int):
    """
    Preprocess the loan data from a DataFrame:
      - numeric features scaled
      - categorical features one-hot
      - target in column 'loan_status'
    """
    df = df.copy()

    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score'
    ]
    categorical_features = [
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]

    # Numeric columns
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Scale numeric
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_features])

    # One-hot encode categorical
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_categorical = ohe.fit_transform(df[categorical_features])

    # Concatenate
    X = np.hstack([X_numeric, X_categorical])

    # Target
    y = df['loan_status'].values.astype(np.int64)
    input_dim = X.shape[1]
    return X, y, input_dim

def load_adult_data(train_file: str, test_file: str, batch_size: int = 512):
    train_df = pd.read_csv(train_file, skipinitialspace=True)
    test_df = pd.read_csv(test_file, skipinitialspace=True)

    X_train, y_train, input_dim = preprocess_loan_data(train_df)
    X_test, y_test, _ = preprocess_loan_data(test_df)

    train_dataset = AdultDataset(X_train, y_train)
    test_dataset = AdultDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader, len(train_dataset), input_dim

# ------------------
# Model
# ------------------

class ImprovedAdultNet(nn.Module):
    def __init__(self, input_dim: int):
        super(ImprovedAdultNet, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(0.1)

        self.fc_out = nn.Linear(64, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.norm3(x)
        x = self.dropout3(x)
        out = self.fc_out(x)
        return out

# ------------------
# DP Training
# ------------------

def train_private_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    privacy_engine,
    epochs: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    target_delta=1e-5
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_model_state = None
    all_train_accs = []
    all_test_accs = []
    all_epsilons = []
    all_train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(output, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if (batch_idx + 1) % 20 == 0:
                current_batch_loss = running_loss / (batch_idx + 1)
                current_train_acc = 100. * correct_train / total_train
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Avg Loss: {current_batch_loss:.3f}, Batch Train Acc: {current_train_acc:.2f}%')

        avg_epoch_loss = epoch_loss / num_batches
        all_train_losses.append(avg_epoch_loss)

        # Evaluate
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        train_acc = 100. * correct_train / total_train
        test_acc = 100. * correct_test / total_test
        epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)

        print(f'Epoch: {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Loss: {avg_epoch_loss:.3f}, ε: {epsilon:.2f}')

        all_train_accs.append(train_acc)
        all_test_accs.append(test_acc)
        all_epsilons.append(epsilon)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return all_train_accs, all_test_accs, all_epsilons, all_train_losses


# ------------------
# Evaluate TPR Gap (Equal Opportunity)
# ------------------

def evaluate_equal_opportunity(model, test_loader, sensitive_values):
    """
    Computes TPR for label=1 across each sensitive group (male/female).
    Returns a dict {group_value: TPR}.
    """
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    sens_vals = np.array(sensitive_values)

    results = {}
    for group in np.unique(sens_vals):
        mask = (sens_vals == group)
        group_true = all_true[mask]
        group_preds = all_preds[mask]
        TP = np.sum((group_true == 1) & (group_preds == 1))
        FN = np.sum((group_true == 1) & (group_preds == 0))
        TPR = TP / (TP + FN) if (TP+FN) > 0 else 0.0
        results[group] = TPR
    return results


# ------------------
# Plotting
# ------------------

def plot_results(train_accs, test_accs, epsilons, train_losses, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_losses, label='Training Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, epsilons, label='ε (Privacy Budget)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('ε')
    plt.title('Privacy Budget over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------
# AIF360 Post-Processing for Equal Opportunity
# ------------------

def aif360_equal_opportunity_postprocess(model, test_loader, test_df, device):
    """
    Use AIF360's CalibratedEqOddsPostprocessing with cost_constraint='fnr'
    to approximate Equal Opportunity. We'll gather predicted probabilities
    from the DP model, then let AIF360 adjust them to reduce TPR gap.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            logits = model(data)
            # Probability of class=1
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels= np.concatenate(all_labels)

    # Convert 'male'/'female' to numeric 0/1
    sens_attr_str = test_df['person_gender'].values
    sens_attr = np.where(sens_attr_str=='male', 1, 0)

    # 1) Construct a minimal DataFrame for AIF360
    import pandas as pd
    df_for_aif = pd.DataFrame({
        "label": all_labels,   # ground truth label
        "gender": sens_attr    # protected attribute
    })

    # 2) Build a BinaryLabelDataset with df=...
    bld_test = BinaryLabelDataset(
        df=df_for_aif,
        label_names=['label'],
        protected_attribute_names=['gender'],
        unprivileged_protected_attributes=[[0]],  # 'female' => 0
        privileged_protected_attributes=[[1]],    # 'male' => 1
        favorable_label=1.0,
        unfavorable_label=0.0
    )

    # 3) We can store predicted probabilities in .scores
    bld_test.scores = all_probs.reshape(-1,1)

    # 4) Initialize CalibratedEqOddsPostprocessing
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
    postproc = CalibratedEqOddsPostprocessing(
        unprivileged_groups=[{'gender': 0}],
        privileged_groups=[{'gender': 1}],
        cost_constraint='fnr',  # tries to align TPR => Equal Opportunity
        seed=42
    )

    # Fit on the same dataset (for demonstration)
    postproc.fit(bld_test, bld_test)

    # Transform => new predictions
    bld_test_transf = postproc.predict(bld_test)
    new_preds = bld_test_transf.labels.ravel()

    # Evaluate TPR gap
    results_after = {}
    for grp in np.unique(sens_attr):
        mask = (sens_attr == grp)
        group_true = all_labels[mask]
        group_preds = new_preds[mask]
        TP = np.sum((group_true == 1) & (group_preds == 1))
        FN = np.sum((group_true == 1) & (group_preds == 0))
        TPR = TP / (TP + FN) if (TP+FN) > 0 else 0.0
        results_after[grp] = TPR

    gap_after = abs(results_after.get(0,0) - results_after.get(1,0))
    acc_after = np.mean(new_preds == all_labels)

    return new_preds, results_after, gap_after, acc_after


def main():
    # Base config
    BASE_SEED = 42
    NUM_RUNS = 1
    run_seeds = generate_run_seeds(BASE_SEED, NUM_RUNS)
    print(f"Generated seeds for runs: {run_seeds}")

    BATCH_SIZE = 5000
    EPOCHS = 11
    LEARNING_RATE = 0.005
    NOISE_MULTIPLIER = 4
    MAX_GRAD_NORM = 1
    TARGET_DELTA = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_train_accuracies = []
    all_test_accuracies = []
    all_epsilons = []
    all_train_losses = []

    # Variables to store the final "raw" gap and "fair" gap:
    raw_gap_final = None
    raw_acc_final = None
    fair_gap_final = None
    fair_acc_final = None

    for run_idx, run_seed in enumerate(run_seeds):
        print(f'\nRun {run_idx + 1}/{NUM_RUNS} (Seed: {run_seed})')
        set_all_seeds(run_seed)

        # 1) Load data
        train_loader, test_loader, sample_size, input_dim = load_adult_data(
            "train.csv", "test.csv", batch_size=BATCH_SIZE
        )

        # 2) Build model
        print("Initializing model...")
        model = ImprovedAdultNet(input_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

        print("Initializing privacy engine...")
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
            poisson_sampling=False
        )

        # 3) Train with DP
        print("Starting DP training...")
        train_accs, test_accs, epsilons, train_losses = train_private_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            privacy_engine=privacy_engine,
            epochs=EPOCHS,
            device=device,
            target_delta=TARGET_DELTA
        )

        all_train_accuracies.append(train_accs)
        all_test_accuracies.append(test_accs)
        all_epsilons.append(epsilons)
        all_train_losses.append(train_losses)

        # (the final test_acc after last epoch is test_accs[-1])
        raw_acc_final = test_accs[-1]  # store the final "raw" accuracy

        # 4) Evaluate TPR gap BEFORE
        test_df = pd.read_csv("test.csv", skipinitialspace=True)
        sensitive_values = test_df['person_gender'].values
        fairness_results_before = evaluate_equal_opportunity(model, test_loader, sensitive_values)
        female_tpr_before = fairness_results_before.get('female', 0.0)
        male_tpr_before   = fairness_results_before.get('male', 0.0)
        gap_before = abs(female_tpr_before - male_tpr_before)

        print("\nFairness (Equal Opportunity) BEFORE mitigation:")
        print(f"  TPR Female: {female_tpr_before:.2f}, TPR Male: {male_tpr_before:.2f}")
        print(f"  TPR Gap BEFORE: {gap_before:.2f}")
        raw_gap_final = gap_before  # store the final "raw" gap

        # 5) AIF360 post-process => reduce TPR gap
        print("\nApplying AIF360 CalibratedEqOddsPostprocessing (cost_constraint='fnr') for Equal Opportunity...")
        new_preds, results_after, gap_after, acc_after = aif360_equal_opportunity_postprocess(
            model, test_loader, test_df, device
        )

        fem_tpr_after = results_after.get(0, 0.0)  # 0 => female
        male_tpr_after= results_after.get(1, 0.0)  # 1 => male

        print("Fairness (Equal Opportunity) AFTER mitigation (AIF360):")
        print(f"  TPR Female: {fem_tpr_after:.2f}, TPR Male: {male_tpr_after:.2f}")
        print(f"  TPR Gap AFTER: {gap_after:.2f}")
        print(f"  Accuracy AFTER mitigation: {acc_after:.2f}")

        fair_gap_final = gap_after   # store the final "fair" gap
        fair_acc_final = acc_after   # store the final "fair" accuracy


    # 6) Summaries across runs
    avg_train_accs = np.mean(all_train_accuracies, axis=0)
    avg_test_accs  = np.mean(all_test_accuracies, axis=0)
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_epsilons   = np.squeeze(np.mean(all_epsilons, axis=0))

    print("\nFinal Results (Averaged Over Runs):")
    print("Average Train Accuracy:", avg_train_accs)
    print("Average Test Accuracy:",  avg_test_accs)
    print("Final Privacy Budget (ε):", avg_epsilons)

    # Example: Print final raw vs. fair metrics from the last run
    print("\n--- Comparison of Raw vs. Post-Processed Fairness ---")
    print(f"Raw Test Accuracy (No fairness fix): {raw_acc_final:.2f}")
    print(f"Raw TPR Gap: {raw_gap_final:.2f}")
    print(f"Fair Test Accuracy (Post-Processed): {fair_acc_final:.2f}")
    print(f"Fair TPR Gap: {fair_gap_final:.2f}")

    plot_results(avg_train_accs, avg_test_accs, avg_epsilons, avg_train_losses, EPOCHS)


if __name__ == "__main__":
    main()
