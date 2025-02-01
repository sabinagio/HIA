def process_csv_to_collection(csv_path, collection):
    import pandas as pd
    import json

    # Read CSV file from wherever
    df = pd.read_csv(csv_path)

    # Validate required column - we need the list of columns and rename/normalize names
    # there should only been 1 text column - all the text columns concatenated to become a document
    column_list = []
    for col in column_list:
        if col not in df.columns:
            raise ValueError(f"CSV must contain a {col} column")

    # Initialize lists for documents and metadata
    documents = []
    metadatas = []

    # Process each row
    for _, row in df.iterrows():
        # Add document text
        documents.append(row['text'])

        # Create metadata dictionary
        metadata = {}

        # Add standard metadata fields if they exist
        for field in ['source', 'last_updated', 'domain']:
            if field in row and pd.notna(row[field]):
                metadata[field] = row[field]

        # Process contact information - this can change based on the actual data
        contact = {}
        if 'email' in row and pd.notna(row['email']):
            contact['email'] = row['email']
        if 'phone' in row and pd.notna(row['phone']):
            contact['phone'] = str(row['phone'])  # Convert to string in case it's numeric

        if contact:
            metadata['contact'] = json.dumps(contact)

        metadatas.append(metadata)

    # Generate IDs
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection