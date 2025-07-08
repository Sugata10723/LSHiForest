import pandas as pd

class DataLoader:
    @staticmethod
    def _bin_numerical_features(df, numerical_columns, n_bins=10):
        df_binned = df.copy()
        for col in numerical_columns:
            unique_vals = df[col].nunique()
            if unique_vals > n_bins:
                try:
                    df_binned[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                except ValueError:
                    df_binned[col] = pd.cut(df[col], bins=n_bins, labels=False, include_lowest=True)
            else:
                df_binned[col] = df[col]
        return df_binned

    @staticmethod
    def _load_and_preprocess_data(config):
        df = pd.read_csv(config['filepath'], names=config.get('column_names'), header=config.get('header'), low_memory=False)
        if config.get('column_names'):
            df.columns = config['column_names']
        
        if config.get('n_rows'):
            df = df.iloc[:config['n_rows'], :]

        if 'label_mapping' in config:
            df[config['label_col']] = df[config['label_col']].apply(lambda x: config['label_mapping'].get(x, 1))

        normal_data = df[df[config['label_col']] == 0]
        anomaly_data = df[df[config['label_col']] == 1]

        normal_data_shuffled = normal_data.sample(frac=1, random_state=42).reset_index(drop=True)
        anomaly_data_shuffled = anomaly_data.sample(frac=1, random_state=42).reset_index(drop=True)

        train_samples = config.get('train_samples', int(len(normal_data_shuffled) * 0.8))
        normal_train = normal_data_shuffled.iloc[:train_samples]
        normal_test = normal_data_shuffled.iloc[train_samples:]

        test_data = pd.concat([normal_test, anomaly_data_shuffled]).sample(frac=1, random_state=42).reset_index(drop=True)

        train_binned = DataLoader._bin_numerical_features(normal_train, config['numerical_features'])
        test_binned = DataLoader._bin_numerical_features(test_data, config['numerical_features'])

        features_to_use = config['categorical_features'] + config['numerical_features']
        
        X_train = train_binned[features_to_use]
        y_train = normal_train[config['label_col']]
        
        test_samples = config.get('test_samples')
        X_test = test_binned[features_to_use].iloc[:test_samples] if test_samples else test_binned[features_to_use]
        y_test = test_binned[config['label_col']].iloc[:test_samples] if test_samples else test_binned[config['label_col']]
        attack_cats = test_binned[config['attack_cat_col']].iloc[:test_samples] if test_samples else test_binned[config['attack_cat_col']]

        return X_train, y_train, X_test, y_test, attack_cats

    @staticmethod
    def load_nsl_kdd(train_samples=50000, test_samples=20000):
        config = {
            'filepath': "data/nsl-kdd/KDDTrain+.txt",
            'column_names': [
                'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
                'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
                'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
                'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
                'dst_host_srv_rerror_rate','label', 'difficulty'
            ],
            'categorical_features': ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'],
            'numerical_features': [],
            'label_col': 'label',
            'attack_cat_col': 'label',
            'label_mapping': {'normal': 0},
            'train_samples': train_samples,
            'test_samples': test_samples
        }
        config['numerical_features'] = [col for col in config['column_names'] if col not in config['categorical_features'] + [config['label_col'], 'difficulty']]
        return DataLoader._load_and_preprocess_data(config)

    @staticmethod
    def load_unsw_nb15(n_rows=10000):
        features_df = pd.read_csv("data/unsw-nb15/NUSW-NB15_features.csv", encoding='cp1252')
        feature_names = features_df['Name'].tolist()
        config = {
            'filepath': "data/unsw-nb15/UNSW-NB15_1.csv",
            'header': None,
            'column_names': feature_names,
            'n_rows': n_rows,
            'categorical_features': ['sport', 'dsport', 'proto', 'service', 'state', 'is_sm_ips_ports', 'is_ftp_login'],
            'numerical_features': [
                "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "Sload", "Dload", "Spkts", "Dpkts",
                "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
                "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "ct_state_ttl",
                "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
                "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
            ],
            'label_col': 'Label',
            'attack_cat_col': 'attack_cat'
        }
        return DataLoader._load_and_preprocess_data(config)
