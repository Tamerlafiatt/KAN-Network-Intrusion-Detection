"""
data_loader.py  —  CICIDS2016 loading, cleaning, splitting
"""
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import warnings; warnings.filterwarnings('ignore')


def normalize_columns(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ','_').str.replace('/','_').str.replace('-','_'))
    return df


def load_cicids2016(data_dir='./data', sample_frac=1.0, random_state=42):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        print('No CSV files found — generating synthetic CICIDS2016-like data...')
        return _generate_synthetic(n_samples=50000, random_state=random_state)
    dfs = []
    for f in sorted(csv_files):
        print(f'  Loading: {os.path.basename(f)}')
        try:    df = pd.read_csv(f, encoding='utf-8',   low_memory=False)
        except: df = pd.read_csv(f, encoding='latin-1', low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = normalize_columns(df)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    print(f'  Loaded {len(df):,} rows x {len(df.columns)} columns')
    return df


def _generate_synthetic(n_samples=50000, random_state=42):
    rng = np.random.RandomState(random_state)
    FEATURES = [
        'destination_port','flow_duration','total_fwd_packets','total_backward_packets',
        'total_length_of_fwd_packets','total_length_of_bwd_packets','fwd_packet_length_max',
        'fwd_packet_length_min','fwd_packet_length_mean','fwd_packet_length_std',
        'bwd_packet_length_max','bwd_packet_length_min','bwd_packet_length_mean',
        'bwd_packet_length_std','flow_bytes_s','flow_packets_s','flow_iat_mean',
        'flow_iat_std','flow_iat_max','flow_iat_min','fwd_iat_total','fwd_iat_mean',
        'fwd_iat_std','fwd_iat_max','fwd_iat_min','bwd_iat_total','bwd_iat_mean',
        'bwd_iat_std','bwd_iat_max','bwd_iat_min','fwd_psh_flags','bwd_psh_flags',
        'fwd_urg_flags','bwd_urg_flags','fwd_header_length','bwd_header_length',
        'fwd_packets_s','bwd_packets_s','min_packet_length','max_packet_length',
        'packet_length_mean','packet_length_std','packet_length_variance',
        'fin_flag_count','syn_flag_count','rst_flag_count','psh_flag_count',
        'ack_flag_count','urg_flag_count','cwe_flag_count','ece_flag_count',
        'down_up_ratio','average_packet_size','avg_fwd_segment_size',
        'avg_bwd_segment_size','fwd_header_length_1','fwd_avg_bytes_bulk',
        'fwd_avg_packets_bulk','fwd_avg_bulk_rate','bwd_avg_bytes_bulk',
        'bwd_avg_packets_bulk','bwd_avg_bulk_rate','subflow_fwd_packets',
        'subflow_fwd_bytes','subflow_bwd_packets','subflow_bwd_bytes',
        'init_win_bytes_forward','init_win_bytes_backward','act_data_pkt_fwd',
        'min_seg_size_forward','active_mean','active_std','active_max','active_min',
        'idle_mean','idle_std','idle_max','idle_min',
    ]
    LABELS  = ['BENIGN','DoS Hulk','PortScan','DDoS','DoS GoldenEye',
               'FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest',
               'Bot','Web Attack Brute Force','Web Attack XSS',
               'Infiltration','Web Attack Sql Injection','Heartbleed']
    weights = np.array([0.55,0.10,0.08,0.07,0.04,0.03,0.03,0.02,0.02,
                        0.02,0.01,0.01,0.01,0.005,0.005])
    weights /= weights.sum()
    labels = rng.choice(LABELS, size=n_samples, p=weights)
    data = {}
    for f in FEATURES:
        if 'flag' in f:
            data[f] = rng.randint(0, 2, n_samples).astype(float)
        elif 'port' in f:
            data[f] = rng.randint(0, 65536, n_samples).astype(float)
        else:
            data[f] = np.abs(rng.exponential(1000, n_samples) + rng.normal(0, 50, n_samples))
    atk = labels != 'BENIGN'
    data['flow_packets_s'][atk] *= rng.uniform(1.5, 5.0, atk.sum())
    data['flow_bytes_s'][atk]   *= rng.uniform(1.2, 4.0, atk.sum())
    df = pd.DataFrame(data)
    df['label'] = labels
    print(f'  Synthetic: {len(df):,} rows, label dist:\n{df.label.value_counts().to_string()}')
    return df


def clean_data(df, label_col='label'):
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in df.columns and label_col not in num_cols:
        num_cols.append(label_col)
    df = df[num_cols].reset_index(drop=True)
    print(f'  Cleaned: {n0:,} -> {len(df):,} rows')
    return df


def encode_labels(df, label_col='label', mode='binary'):
    df = df.copy()
    if mode == 'binary':
        df[label_col] = df[label_col].astype(str).str.strip().str.upper()
        df['encoded_label'] = (df[label_col] != 'BENIGN').astype(int)
        n_classes, le = 2, None
        print(f'  Binary: {df.encoded_label.value_counts().to_dict()}')
    else:
        le = LabelEncoder()
        df['encoded_label'] = le.fit_transform(df[label_col].astype(str).str.strip())
        n_classes = len(le.classes_)
        print(f'  Multi-class: {n_classes} classes')
    return df.drop(columns=[label_col]), le, n_classes


def prepare_data(data_dir='./data', label_col='label', mode='binary',
                 val_size=0.15, test_size=0.15, imbalance_strategy='smote',
                 sample_frac=1.0, random_state=42):
    print('\n' + '='*60 + '\n  DATA PIPELINE\n' + '='*60)
    df = load_cicids2016(data_dir, sample_frac, random_state)
    df = normalize_columns(df)
    for col in ['label',' label','labels','class','attack_type']:
        if col in df.columns: label_col = col; break
    df = clean_data(df, label_col)
    df, le, n_classes = encode_labels(df, label_col, mode)
    y = df['encoded_label'].values
    X = df.drop(columns=['encoded_label']).values.astype(np.float32)
    feature_names = df.drop(columns=['encoded_label']).columns.tolist()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size+test_size, stratify=y, random_state=random_state)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size/(val_size+test_size),
        stratify=y_tmp, random_state=random_state)
    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)
    if imbalance_strategy == 'smote':
        u, c = np.unique(y_tr, return_counts=True)
        mn = c.min()
        if mn >= 6:
            sm = SMOTE(random_state=random_state, k_neighbors=min(5, mn-1))
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            print(f'  SMOTE applied -> {len(X_tr):,} train samples')
    print(f'  Split: train={len(X_tr):,} val={len(X_val):,} test={len(X_te):,}')
    print('='*60 + '\n')
    return dict(X_train=X_tr.astype(np.float32), X_val=X_val.astype(np.float32),
                X_test=X_te.astype(np.float32), y_train=y_tr.astype(np.int64),
                y_val=y_val.astype(np.int64), y_test=y_te.astype(np.int64),
                n_features=X_tr.shape[1], n_classes=n_classes,
                label_encoder=le, scaler=sc, feature_names=feature_names)
