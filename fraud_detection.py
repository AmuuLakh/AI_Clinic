import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import time
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """
    Enhanced Fraud Detection Model with Feature Analysis
    """
    
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.feature_importance_df = None
        self.selected_features = None
        
    def load_data(self, file_path, sample_size=20000):
        """Load data sample"""
        print(f"Loading {sample_size} rows from dataset...")
        
        try:
            total_rows = sum(1 for line in open(file_path)) - 1
            sample_frac = sample_size / total_rows if total_rows > sample_size else 1.0
            
            if sample_frac < 1.0:
                df = pd.read_csv(file_path, skiprows=lambda i: i>0 and np.random.random() > sample_frac)
            else:
                df = pd.read_csv(file_path)
            
            print(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try alternative loading method
            df = pd.read_csv(file_path, nrows=sample_size)
            print(f"Loaded {len(df)} rows using alternative method")
            return df
    
    def auto_detect_feature_types(self, df):
        """Automatically detect feature types"""
        numeric_features = []
        categorical_features = []
        
        for col in df.columns:
            if col == 'isFraud':
                continue
                
            # Skip non-numeric columns that are likely identifiers
            if col in ['Date', 'nameOrig', 'nameDest']:
                continue
                
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)
                
        return numeric_features, categorical_features
    
    def create_advanced_features(self, df):
        """Create advanced features for fraud detection"""
        print("Creating advanced features...")
        
        # Basic date features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['transaction_hour'] = df['Date'].dt.hour
            df['transaction_day'] = df['Date'].dt.day
            df['transaction_month'] = df['Date'].dt.month
            df['transaction_year'] = df['Date'].dt.year
            df['is_weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
        
        # Frequency encoding for categorical variables
        for col in ['City', 'type', 'Card Type', 'Exp Type', 'Gender']:
            if col in df.columns:
                freq_map = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
        
        # Balance features
        if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig']):
            df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            df['balance_change_abs'] = abs(df['balance_change'])
            df['balance_change_ratio'] = np.where(
                df['oldbalanceOrg'] > 0, 
                df['balance_change'] / df['oldbalanceOrg'], 
                0
            )
        
        # Transaction patterns
        if 'amount' in df.columns:
            df['amount_to_balance_ratio'] = np.where(
                df['oldbalanceOrg'] > 0,
                df['amount'] / df['oldbalanceOrg'],
                0
            )
            df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # Time-based features (if transaction ordering is available)
        if 'Date' in df.columns:
            df = df.sort_values('Date')
            df['time_since_last_tx'] = df['Date'].diff().dt.total_seconds().fillna(0)
        
        return df
    
    def analyze_feature_importance(self, X, y):
        """Analyze which features are most important"""
        print("\nAnalyzing feature importance...")
        
        # Quick Random Forest to get initial feature importance
        temp_rf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Preprocess for feature importance analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                # For categorical, we get one column per category
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cols)
                feature_names.extend(cat_features)
        
        # Fit model and get importance
        temp_rf.fit(X_processed, y)
        
        # Create feature importance dataframe
        importance_scores = temp_rf.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        top_features = self.feature_importance_df.head(10)
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.feature_importance_df
    
    def select_best_features(self, X, y, importance_threshold=0.01):
        """Select only the most important features"""
        if self.feature_importance_df is None:
            self.analyze_feature_importance(X, y)
        
        # Select features above importance threshold
        important_features = self.feature_importance_df[
            self.feature_importance_df['importance'] > importance_threshold
        ]['feature'].tolist()
        
        # Map back to original column names for categorical features
        original_important_features = []
        for feat in important_features:
            # For one-hot encoded features, find the original column
            if '__' in feat:  # sklearn OneHotEncoder format
                original_col = feat.split('__')[0]
                if original_col not in original_important_features and original_col in X.columns:
                    original_important_features.append(original_col)
            else:
                if feat in X.columns:
                    original_important_features.append(feat)
        
        # Ensure we have at least some features
        if not original_important_features:
            print("No features above threshold, using top 5 features")
            top_original = []
            for feat in self.feature_importance_df.head(5)['feature']:
                if '__' in feat:
                    original_col = feat.split('__')[0]
                    if original_col not in top_original and original_col in X.columns:
                        top_original.append(original_col)
                else:
                    if feat in X.columns:
                        top_original.append(feat)
            original_important_features = top_original
        
        print(f"\nSelected {len(original_important_features)} important features:")
        for feat in original_important_features:
            print(f"  - {feat}")
        
        self.selected_features = original_important_features
        return X[original_important_features]
    
    def preprocess_data(self, df):
        """Enhanced preprocessing with feature engineering"""
        print("\n1. PREPROCESSING DATA...")
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Remove original date column if it exists (we've extracted features from it)
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        
        # Remove identifier columns
        df = df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')
        
        # Define target
        target_col = 'isFraud'
        
        # Remove missing targets
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].astype(int)
        
        # Store info
        self.model_info['dataset'] = {
            'final_shape': X.shape,
            'fraud_cases': y.sum(),
            'fraud_percentage': f"{y.mean()*100:.2f}%",
            'original_features': X.columns.tolist()
        }

        print(f"Original features: {X.shape[1]}")
        print(f"Final dataset: {X.shape}")
        print(f"Fraud rate: {y.sum()} cases ({y.mean()*100:.2f}%)")
        
        # Analyze and select important features
        self.analyze_feature_importance(X, y)
        X_selected = self.select_best_features(X, y)
        
        self.model_info['dataset']['selected_features'] = self.selected_features
        self.model_info['dataset']['selected_shape'] = X_selected.shape
        
        print(f"After feature selection: {X_selected.shape}")
        
        return X_selected, y
    
    def train_model(self, X, y):
        """Train model with feature selection"""
        print("\n2. TRAINING MODEL...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Auto-detect feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        # Define preprocessing
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        # Create model with feature selection
        self.model = Pipeline([
            ('pre', preprocessor),
            ('feature_selector', SelectFromModel(
                RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    class_weight='balanced'
                ), 
                threshold='median'
            )),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=50,
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            ))
        ])

        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        # Store feature information
        try:
            selected_mask = self.model.named_steps['feature_selector'].get_support()
            feature_names_after_preprocessing = (
                numeric_features + 
                list(self.model.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(categorical_features))
            )
            selected_features = [feature_names_after_preprocessing[i] for i in range(len(selected_mask)) if selected_mask[i]]
            self.model_info['final_features_used'] = selected_features
            print(f"Final features used by model: {len(selected_features)}")
        except Exception as e:
            print(f"Could not extract final feature information: {e}")
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Enhanced evaluation with feature analysis"""
        print("\n3. EVALUATING MODEL...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:,1]

        # Classification report
        print("\nCLASSIFICATION REPORT:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        formatted_report = []
        for idx, row in report_df.iterrows():
            if idx in ['0', '1']:
                label = 'NOT FRAUD' if idx == '0' else 'FRAUD'
                formatted_report.append([
                    label,
                    f"{row['precision']:.1%}",
                    f"{row['recall']:.1%}", 
                    f"{row['f1-score']:.1%}",
                    f"{int(row['support']):,}"
                ])

        print(tabulate(formatted_report, 
                       headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                       tablefmt='grid'))

        # ROC AUC and confusion matrix
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        cm_percentage = cm / cm.sum() * 100

        print("\nCONFUSION MATRIX (Counts | Percentages):")
        print("               Predicted NOT FRAUD   Predicted FRAUD")
        print("               -------------------   --------------")
        print(f"Actual NOT FRAUD: {cm[0,0]:>6} | {cm_percentage[0,0]:>5.1f}%     {cm[0,1]:>6} | {cm_percentage[0,1]:>5.1f}%")
        print(f"Actual FRAUD:     {cm[1,0]:>6} | {cm_percentage[1,0]:>5.1f}%     {cm[1,1]:>6} | {cm_percentage[1,1]:>5.1f}%")
        
        # Store results
        self.model_info['evaluation'] = {
            'roc_auc': round(roc_auc, 4),
            'confusion_matrix': cm.tolist(),
            'feature_importance': self.feature_importance_df.to_dict('records') if self.feature_importance_df is not None else []
        }
        
        return y_pred, y_proba
    
    def generate_feature_analysis_report(self):
        """Generate detailed feature analysis report"""
        if self.feature_importance_df is not None:
            print("\nFEATURE IMPORTANCE ANALYSIS:")
            feature_table = []
            for _, row in self.feature_importance_df.head(15).iterrows():
                feature_table.append([
                    row['feature'],
                    f"{row['importance']:.4f}",
                    "✓" if row['importance'] > 0.01 else "✗"
                ])
            
            print(tabulate(feature_table, 
                          headers=['Feature', 'Importance', 'Useful (>0.01)'],
                          tablefmt='grid'))
            
            # Save feature analysis
            self.feature_importance_df.to_csv('feature_importance_analysis.csv', index=False)
            print(f"\nFeature importance analysis saved to 'feature_importance_analysis.csv'")
    
    def generate_predictions_report(self, X_test, y_test, y_pred, y_proba, df):
        """Generate predictions report"""
        print("\n4. GENERATING PREDICTIONS REPORT...")

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Transaction_ID': X_test.index,
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Fraud_Probability': y_proba,
            'Amount': df.loc[X_test.index, 'amount'].values if 'amount' in df.columns else np.nan,
            'Type': df.loc[X_test.index, 'type'].values if 'type' in df.columns else 'Unknown'
        })

        # Add status and risk level
        predictions_df['Status'] = np.where(
            predictions_df['Actual'] == predictions_df['Predicted'], 'CORRECT', 'WRONG'
        )

        def get_risk_level(prob):
            if prob < 0.3: return 'LOW RISK'
            elif prob < 0.7: return 'MEDIUM RISK'
            else: return 'HIGH RISK'

        predictions_df['Risk_Level'] = predictions_df['Fraud_Probability'].apply(get_risk_level)

        # Display sample predictions
        print(f"\nSAMPLE PREDICTIONS (First 10 transactions):")
        sample_table = []
        for _, row in predictions_df.head(10).iterrows():
            status_icon = '✓' if row['Status'] == 'CORRECT' else '✗'
            actual_label = 'FRAUD' if row['Actual'] == 1 else 'NOT FRAUD'
            predicted_label = 'FRAUD' if row['Predicted'] == 1 else 'NOT FRAUD'
            
            sample_table.append([
                row['Transaction_ID'],
                f"${row['Amount']:,.2f}" if 'amount' in df.columns else "N/A",
                row['Type'],
                actual_label,
                predicted_label,
                f"{row['Fraud_Probability']:.1%}",
                row['Risk_Level'],
                f"{status_icon} {row['Status']}"
            ])

        print(tabulate(sample_table, 
                       headers=['ID', 'Amount', 'Type', 'Actual', 'Predicted', 'Prob %', 'Risk Level', 'Status'],
                       tablefmt='grid'))

        # Save report
        predictions_df.to_csv('detailed_predictions_report.csv', index=False)
        print(f"\nDetailed predictions saved to 'detailed_predictions_report.csv'")
        
        return predictions_df
    
    def save_model(self, filename="fraud_detection_model.joblib"):
        """Save the model as .joblib"""
        print(f"\n5. SAVING MODEL AS {filename}...")
        
        # Create model package
        model_package = {
            'model': self.model,
            'model_info': self.model_info,
            'feature_importance': self.feature_importance_df,
            'selected_features': self.selected_features,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save using joblib
        joblib.dump(model_package, filename)
        
        print(f"Model saved as '{filename}'")
        
        # Display model info
        print("\nMODEL INFORMATION:")
        info_table = [
            ["Model Type", "RandomForestClassifier with Feature Selection"],
            ["Training Date", model_package['timestamp']],
            ["Original Features", f"{self.model_info['dataset']['final_shape'][1]:,}"],
            ["Selected Features", f"{len(self.selected_features) if self.selected_features else 'N/A':,}"],
            ["Dataset Size", f"{self.model_info['dataset']['final_shape'][0]:,} transactions"],
            ["Fraud Rate", self.model_info['dataset']['fraud_percentage']],
            ["ROC AUC Score", f"{self.model_info['evaluation']['roc_auc']:.4f}"]
        ]
        print(tabulate(info_table, tablefmt='grid'))
    
    def run_pipeline(self, data_path):
        """Run the complete enhanced pipeline"""
        start_time = time.time()
        
        print("=" * 60)
        print("ENHANCED FRAUD DETECTION PIPELINE WITH FEATURE ANALYSIS")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_data(data_path)
        X, y = self.preprocess_data(df)
        
        # Train and evaluate
        X_test, y_test = self.train_model(X, y)
        y_pred, y_proba = self.evaluate_model(X_test, y_test)
        
        # Generate feature analysis
        self.generate_feature_analysis_report()
        
        # Generate report
        predictions_df = self.generate_predictions_report(X_test, y_test, y_pred, y_proba, df)
        
        # Save model
        self.save_model("enhanced_fraud_detection_model.joblib")
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed in {total_time:.2f} seconds!")
        
        return predictions_df

# Usage
if __name__ == "__main__":
    fraud_model = FraudDetectionModel()
    predictions = fraud_model.run_pipeline('Fraud.csv')
    
    # Final sample display
    print(f"\nFINAL SAMPLE PREDICTIONS:")
    final_sample = predictions.head(5)
    for _, row in final_sample.iterrows():
        status_icon = '✓' if row['Status'] == 'CORRECT' else '✗'
        actual_label = 'FRAUD' if row['Actual'] == 1 else 'NOT FRAUD'
        predicted_label = 'FRAUD' if row['Predicted'] == 1 else 'NOT FRAUD'
        
        print(f"{status_icon} TX {row['Transaction_ID']:>6} | "
              f"Amount: ${row['Amount']:>10,.2f} | "
              f"Actual: {actual_label:>10} | "
              f"Predicted: {predicted_label:>10} | "
              f"Probability: {row['Fraud_Probability']:.1%} | "
              f"Risk: {row['Risk_Level']}")