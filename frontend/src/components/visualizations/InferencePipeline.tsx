import {
  ArrowRight,
  SlidersHorizontal,
  Workflow,
  Layers3,
  FileCheck2,
} from "lucide-react";
const stages = [
  {
    label: "Traffic Features",
    detail: "Dataset-specific inputs",
    Icon: SlidersHorizontal,
  },
  { label: "Preprocessing", detail: "Fitted transformations", Icon: Workflow },
  { label: "CNN", detail: "TrafficCNN-1D inference", Icon: Layers3 },
  {
    label: "Classification Result",
    detail: "Class + softmax scores",
    Icon: FileCheck2,
  },
];
export function InferencePipeline() {
  return (
    <section className="inference-strip" aria-label="CNN inference pipeline">
      <div className="inference-title">
        <h2>Classification pipeline</h2>
        <span>Supplied feature records</span>
      </div>
      <ol>
        {stages.map(({ label, detail, Icon }, index) => (
          <li key={label}>
            <span className="pipeline-icon">
              <Icon size={20} strokeWidth={1.5} />
            </span>
            <div>
              <strong>{label}</strong>
              <small>{detail}</small>
            </div>
            {index < stages.length - 1 && (
              <ArrowRight className="stage-arrow" size={18} strokeWidth={1.4} />
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
