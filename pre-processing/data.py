try:
    train_df = pd.read_json('train.jsonl', lines=True)
    test_df = pd.read_json('test.jsonl', lines=True)
    
    if 'player_won' in train_df.columns:
        train_df['player_won'] = train_df['player_won'].astype(int)
    print("Loading successful!")
except Exception as e:
    print(f"Error: {e}")

line_to_remove = 4877

if line_to_remove in train_df.index:
    train_df = train_df.drop(line_to_remove)
    print(f"line {line_to_remove} successfully removed.")
else:
    print(f"line {line_to_remove} not found")

filtro_livello_100 = train_df['p1_team_details'].apply(
    lambda team_list: all(pokemon.get('level') == 100 for pokemon in team_list)
)

train_df = train_df[filtro_livello_100]

def create_expandable_data_viewer(train_df, test_df=None):
    """Create an interactive data viewer with expandable rows."""
    
    display(Markdown("## 📊 Dataset Overview"))
    
    shape_data = {
        'Dataset': ['Training', 'Test'],
        'Rows': [train_df.shape[0], test_df.shape[0] if test_df is not None else 'N/A'],
        'Columns': [train_df.shape[1], test_df.shape[1] if test_df is not None else 'N/A']
    }
    shape_df = pd.DataFrame(shape_data)
    display(shape_df.style.set_caption("Dataset Dimensions"))
    
    display(Markdown("### 📋 Sample Table (First 8 Rows)"))
    sample_df = train_df.head(8).reset_index(drop=True)
    
    styled_sample = sample_df.style\
        .set_table_styles([
            {'selector': 'thead th', 
             'props': [
                 ('background-color', '#2E86AB'), 
                 ('color', 'white'),
                 ('font-weight', 'bold'),
                 ('font-size', '12px'),
                 ('padding', '8px 12px'),
                 ('border', '1px solid #ddd')
             ]},
            {'selector': 'td', 
             'props': [
                 ('font-size', '11px'),
                 ('padding', '6px 10px'),
                 ('border', '1px solid #ddd'),
                 ('max-width', '200px'),
             ]},
            {'selector': 'tbody tr:nth-child(even)', 
             'props': [('background-color', '#f2f2f2')]},
        ])\
        .format({
            col: lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
            for col in sample_df.columns
        })
    
    display(styled_sample)
    display(Markdown("*💡 Use the sliders above to explore individual cell values*"))

create_expandable_data_viewer(train_df, test_df)