import { motion, useReducedMotion } from "motion/react";
import { ArrowRight, Database, Layers3, ScanLine, Workflow, type LucideIcon } from "lucide-react";
import "./about.css";

const technologies: { label: string; Icon: LucideIcon }[] = [
  { label: "1D Convolutional Neural Network (CNN)", Icon: Layers3 },
  { label: "UNSW-NB15 dataset", Icon: Database },
  { label: "ISCX VPN-nonVPN dataset", Icon: Database },
  { label: "Single-record and batch traffic classification", Icon: ScanLine },
];
const steps = [
  { label: "Dataset", Icon: Database },
  { label: "Preprocessing", Icon: Workflow },
  { label: "CNN Model", Icon: Layers3 },
  { label: "Traffic Classification", Icon: ScanLine },
];

export function AboutPage() {
  const reduced = useReducedMotion();
  return (
    <div className="about-product">
      <section className="about-product-intro" aria-labelledby="about-title">
        <span className="about-product-kicker">THE PROJECT</span>
        <h1 id="about-title">About TrafficFlow</h1>
        <p className="about-product-subtitle">CNN-Based Traffic Flow Classification in 5G Core Networks.</p>
        <p className="about-product-description">
          TrafficFlow turns network-flow records into understandable traffic categories using trained CNN models.
          Load a dataset sample or supply your own features to explore how machine learning identifies patterns in network traffic.
        </p>
      </section>

      <section className="about-product-section" aria-labelledby="core-technology-title">
        <h2 id="core-technology-title">Core Technology</h2>
        <ul className="about-technology-list">
          {technologies.map(({ label, Icon }) => (
            <li key={label}><Icon size={22} strokeWidth={1.5} aria-hidden="true" /><span>{label}</span></li>
          ))}
        </ul>
      </section>

      <section className="about-product-section" aria-labelledby="how-it-works-title">
        <h2 id="how-it-works-title">How It Works</h2>
        <ol className="about-process">
          {steps.map(({ label, Icon }, index) => (
            <motion.li
              key={label}
              initial={{ opacity: reduced ? 1 : 0, y: reduced ? 0 : 6 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: reduced ? 0 : 0.3, delay: reduced ? 0 : index * 0.06 }}
            >
              <div className="about-process-top"><span className="mono">0{index + 1}</span><Icon size={23} strokeWidth={1.5} aria-hidden="true" /></div>
              <h3>{label}</h3>
              {index < steps.length - 1 && <ArrowRight className="about-process-arrow" size={18} aria-hidden="true" />}
            </motion.li>
          ))}
        </ol>
      </section>

      <section className="about-product-scope" aria-labelledby="project-scope-title">
        <h2 id="project-scope-title">Project Scope</h2>
        <p>This application classifies supplied network-flow records using trained CNN models. It does not currently capture or classify live network traffic.</p>
      </section>

    </div>
  );
}
