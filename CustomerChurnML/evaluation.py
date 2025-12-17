def business_summary(df):
    summary = {
        "Total Customers": len(df),
        "Saveable Customers": (df["Segment"] == "Saveable").sum(),
        "Not Worth Saving": (df["Segment"] == "Not Worth Saving").sum(),
        "Loyal Customers": (df["Segment"] == "Loyal").sum(),
        "Total Expected Net Gain": round(
            df[df["Segment"] == "Saveable"]["NetGain"].sum(), 2
        )
    }
    return summary
