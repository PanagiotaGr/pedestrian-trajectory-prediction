import pandas as pd

# load clean dataset
df = pd.read_csv("results/dataset_final_clean.csv")

# save as Excel
output_path = "results/dataset_final_clean.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Dataset")

    ws = writer.sheets["Dataset"]

    # auto column width
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter

        for cell in col:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass

        ws.column_dimensions[col_letter].width = max_length + 2

print("Excel saved:", output_path)
