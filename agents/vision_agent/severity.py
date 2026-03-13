def classify_severity(heatmap):

    severity = {}

    rows = "ABCDEFGHIJ"

    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):

            score = heatmap[i][j]

            zone = f"{rows[i]}{j+1}"

            if score > 0.7:
                severity[zone] = "Critical"

            elif score > 0.3:
                severity[zone] = "Moderate"

            else:
                severity[zone] = "Low"

    return severity