def class_labels(column):
    """
    Takes in target column and creates list of binary values. 1 (>=0.5) being in the 
    positive class (toxic), 0 (<0.5) being in the negative class (Not toxic)
    """
    
    class_label = []
    
    for row in column:
        
        if row < 0.5:
            class_label.append(0)
        else:
            class_label.append(1)
            
    return class_label

def clean_text(df, text):
    """
    Cleans text by replacing unwanted characters with blanks
    Replaces @ signs with word at
    Makes all text lowercase
    """
    
    df[text] = df[text].str.replace(r'[^A-Za-z0-9()!?@\s\'\`\*\"\_\n]', '', regex=True)
    df[text] = df[text].str.replace(r'@', 'at', regex=True)
    df[text] = df[text].str.lower()
    
    return df