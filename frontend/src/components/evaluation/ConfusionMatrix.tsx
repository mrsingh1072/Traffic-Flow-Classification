import { useState } from "react";
import type { ConfusionMatrixResponse } from "../../services/api";

export function ConfusionMatrix({ data }: { data: ConfusionMatrixResponse }) {
  const [detail, setDetail] = useState("");
  const max = Math.max(0, ...data.matrix.flat());
  if (!data.labels.length || !data.matrix.length)
    return (
      <p className="empty-message">No confusion matrix values returned.</p>
    );

  return (
    <>
      <div className="matrix-axis">
        Rows: actual class · Columns: predicted class
      </div>
      <div className="table-scroll">
        <table
          className="matrix-table"
          aria-label={data.dataset + " confusion matrix"}
        >
          <thead>
            <tr>
              <th scope="col">Actual / Predicted</th>
              {data.labels.map((label) => (
                <th scope="col" key={label}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.matrix.map((row, ri) => (
              <tr key={ri}>
                <th scope="row">{data.labels[ri]}</th>
                {row.map((value, ci) => {
                  const t = max ? value / max : 0;
                  const rgb = [
                    Math.round(25 + t * 184),
                    Math.round(25 + t * 81),
                    Math.round(25 + t * 57),
                  ];
                  const linear = rgb.map((v) => {
                    const n = v / 255;
                    return n <= 0.04045
                      ? n / 12.92
                      : ((n + 0.055) / 1.055) ** 2.4;
                  });
                  const luminance =
                    linear[0] * 0.2126 +
                    linear[1] * 0.7152 +
                    linear[2] * 0.0722;
                  const label = `Actual ${data.labels[ri]}, predicted ${data.labels[ci]}: ${value.toLocaleString()} samples`;
                  return (
                    <td
                      key={ci}
                      tabIndex={0}
                      aria-label={label}
                      title={label}
                      onFocus={() => setDetail(label)}
                      onMouseEnter={() => setDetail(label)}
                      style={{
                        background: `rgb(${rgb.join(",")})`,
                        color: luminance < 0.179 ? "#FFFFFF" : "#000000",
                        boxShadow:
                          ri === ci ? "inset 0 0 0 1px #686A6588" : undefined,
                      }}
                    >
                      {value === 0 ? "·" : value.toLocaleString()}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="matrix-detail" aria-live="polite">
        {detail ||
          "Focus or hover a cell to inspect the actual and predicted classes."}
      </div>
      <div className="matrix-legend">
        <span>0 samples</span>
        <span className="matrix-scale" />
        <span>{max.toLocaleString()} samples</span>
        <span>· Outlined diagonal = correct classifications</span>
      </div>
    </>
  );
}
