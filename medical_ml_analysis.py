import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def read_table(path):
    df = pd.read_csv(path)
    return df

def transform_target(df):
    df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
    return df

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
    return train_df, test_df
    
def predict_probabilities(train_df, test_df):
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    X_test = test_df.drop('target', axis=1)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    probs_all_classes = rf_model.predict_proba(X_test)
    
    return probs_all_classes[:, 1]

def paint_PCA(df):
    X = df.drop('target', axis=1) 
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Dataset')
    plt.colorbar(scatter, label='Target')
    plt.show()

def paint_histograms(df):

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()  
    
    features = [col for col in df.columns if col != 'target']
    
    for i, feature in enumerate(features):
        sns.histplot(data=df[df['target'] == 0], x=feature, kde=False, 
                     ax=axes[i], color='skyblue', label='No Disease (0)', stat="count", alpha=0.6)
        
        sns.histplot(data=df[df['target'] == 1], x=feature, kde=False, 
                     ax=axes[i], color='salmon', label='Disease (1)', stat="count", alpha=0.6)
        
        axes[i].set_title(f'{feature}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('') 
        axes[i].legend(loc='upper right')
    
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

def paint_ROC_curve(train_df, test_df):

    y_test = test_df['target'] 
    y_scores = predict_probabilities(train_df, test_df) 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig('roc_curve.png')
    plt.show()
    print(f"The AUC score is: {roc_auc}")

def sort_features(train_df, test_df):

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    rf_model = RandomForestClassifier(random_state = 42)
    rf_model.fit(X_train, y_train)
    
    importances = rf_model.feature_importances_
    
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index().drop("index", axis=1)
    return(top_features)

if __name__ == '__main__':
    pass
