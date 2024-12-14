import pandas as pd
import numpy as np

def extract_source_destination_port(text):
    match = re.search(r'(\d+)\s*>\s*(\d+)', text)
    if match:
        source, destination = match.groups()
        # Remove the matched substring from text
        updated_text = text.replace(match.group(0), '', 1).strip()
        return int(source), int(destination), updated_text
    else:
        return np.nan, np.nan, text

def extract_seq(text):
    # Adjust regex to match the sequence number
    match = re.search(r'seq\s*=\s*(\d+)', text, re.IGNORECASE)  #
    if match:
        Seq = match.group(1)  # Get the first group directly
        updated_text = text.replace(match.group(0), '', 1).strip()
        return int(Seq), updated_text
    else:
        return np.nan, text


def extract_transaction_id(text):
    # Regex to match transaction ID starting with 0x and followed by hexadecimal characters
    match = re.search(r'0x[0-9a-fA-F]+', text)
    
    if match:
        # Extract the transaction ID as a string
        transaction_id = match.group(0)
        
        # Remove the transaction ID from the original text
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        
        # Return the integer value of the transaction ID and the updated text
        return int(transaction_id, 16), updated_text  # Convert hex to integer and return
        
    else:
        # If no transaction ID is found, return np.nan for both values
        return np.nan, text



def extract_len(text):
    match = re.search(r'Len=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text


def extract_win(text):
    match = re.search(r'Win=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_MSS(text):
    match = re.search(r'MSS=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_Ack(text):
    match = re.search(r'Ack=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_TSval(text):
    match = re.search(r'TSval=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_TSecr(text):
    match = re.search(r'TSecr=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text


def extract_WS(text):
    match = re.search(r'WS=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_Urg(text):
    match = re.search(r'Urg=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text


def extract_SACK_PERM(text):
    match = re.search(r'SACK_PERM=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text

def extract_ttl(text):
    match = re.search(r'ttl=(\d+)', text)
    if match:
        length = match.group(1) 
        updated_text = text.replace(match.group(0), '').strip()  # Strip to clean up any leading/trailing spaces
        return int(length), updated_text
    else:
        return np.nan, text


def extract_seq_ack(text):
    # Regular expression to match the sequence number and acknowledgment number
    match = re.search(r'TCP Dup ACK (\d+)#(\d+)', text)
    
    if match:
        seq_num = match.group(1)  # Sequence number
        ack_num = match.group(2)  # Acknowledgment number
        updated_text = text.replace(match.group(0), '').strip()  # Clean up the text after extraction
        return int(seq_num), int(ack_num), updated_text
    else:
        return np.nan, np.nan, text

def extract_ips_from_parts(text):
    # Regular expression to match IPv4 addresses
    match = re.search(r'Who has (\b(?:\d{1,3}\.){3}\d{1,3}\b).*?Tell (\b(?:\d{1,3}\.){3}\d{1,3}\b)', text)
    
    if match:
        # Extract the "Who has" IP and "Tell" IP
        who_has_ip = match.group(1)
        tell_ip = match.group(2)
        
        # Clean up the text after extracting the IPs
        updated_text = text.replace(match.group(0), '').strip()
        
        return who_has_ip, tell_ip, updated_text
    else:
        # If no match found, return np.nan for both IPs and the original text
        return np.nan, np.nan, text


def extract_features_from_info(df: pd.DataFrame) -> pd.DataFrame:
    def extract_all_features(text):
        # Apply each extraction function in sequence
        features = {}
        
        # Extracting the source and destination port
        source, destination, updated_text = extract_source_destination_port(text)
        features['source_port'] = source
        features['destination_port'] = destination
        
        # Extracting the sequence number
        seq, updated_text = extract_seq(updated_text)
        features['seq'] = seq
        
        # Extracting the transaction ID
        transaction_id, updated_text = extract_transaction_id(updated_text)
        features['transaction_id'] = transaction_id
        
        # Extracting the length (Len)
        length, updated_text = extract_len(updated_text)
        features['length'] = length
        
        # Extracting the window size (Win)
        win, updated_text = extract_win(updated_text)
        features['win'] = win
        
        # Extracting MSS
        mss, updated_text = extract_MSS(updated_text)
        features['mss'] = mss
        
        # Extracting Acknowledgment number (Ack)
        ack, updated_text = extract_Ack(updated_text)
        features['ack'] = ack
        
        # Extracting timestamp value (TSval)
        tsval, updated_text = extract_TSval(updated_text)
        features['tsval'] = tsval
        
        # Extracting timestamp echo reply (TSecr)
        tsecr, updated_text = extract_TSecr(updated_text)
        features['tsecr'] = tsecr
        
        # Extracting the window scale (WS)
        ws, updated_text = extract_WS(updated_text)
        features['ws'] = ws
        
        # Extracting the urgent pointer (Urg)
        urg, updated_text = extract_Urg(updated_text)
        features['urg'] = urg
        
        # Extracting the SACK permission (SACK_PERM)
        sack_perm, updated_text = extract_SACK_PERM(updated_text)
        features['sack_perm'] = sack_perm
        
        # Extracting TTL
        ttl, updated_text = extract_ttl(updated_text)
        features['ttl'] = ttl
        
        # Extracting duplicate ACK sequence and acknowledgment
        seq_ack, ack_ack, updated_text = extract_seq_ack(updated_text)
        features['seq_ack'] = seq_ack
        features['ack_num'] = ack_ack
        
        # Extracting IPs (Who has and Tell)
        who_has_ip, tell_ip, updated_text = extract_ips_from_parts(updated_text)
        features['who_has_ip'] = who_has_ip
        features['tell_ip'] = tell_ip
        
        # Return all features in a dictionary
        features['rest_of_info'] = updated_text
        return pd.Series(features)
    
    # Apply the extraction to the 'updated_info' column and expand into multiple columns
    df_extracted = df['Info'].apply(extract_all_features)
    
    # Combine the new columns with the original dataframe
    df_combined = pd.concat([df, df_extracted], axis=1)
    
    return df_combined



