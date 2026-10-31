import { motion, useReducedMotion } from "motion/react";
import { Heart, Icon, type IconNode } from "lucide-react";
import "./creator-signature.css";

const socialLinks: { label: string; href: string; icon: IconNode }[] = [
  {
    label: "GitHub", href: "https://github.com/mrsingh1072",
    icon: [["path", { d: "M9 19c-4.3 1.4-4.3-2.5-6-3m12 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 19 4.77 5.07 5.07 0 0 0 18.91 1S17.73.65 15 2.48a13.38 13.38 0 0 0-7 0C5.27.65 4.09 1 4.09 1A5.07 5.07 0 0 0 4 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 8 18.13V22", key: "github" }]],
  },
  {
    label: "LinkedIn", href: "https://www.linkedin.com/in/saurabh-singh-959b48323/",
    icon: [["path", { d: "M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z", key: "in" }], ["rect", { x: 2, y: 9, width: 4, height: 12, key: "stem" }], ["circle", { cx: 4, cy: 4, r: 2, key: "dot" }]],
  },
  {
    label: "Instagram", href: "https://www.instagram.com/mr__singh1072/",
    icon: [["rect", { x: 2, y: 2, width: 20, height: 20, rx: 5, key: "frame" }], ["circle", { cx: 12, cy: 12, r: 4, key: "lens" }], ["path", { d: "M17.5 6.5h.01", key: "flash" }]],
  },
];

export function CreatorSignature() {
  const reduced = useReducedMotion();
  return (
      <section className="about-creator" aria-labelledby="creator-title">
        <h2 id="creator-title">
          Made with <motion.span
            className="creator-heart"
            animate={reduced ? { scale: 1 } : { scale: [1, 1.12, 1, 1.07, 1] }}
            transition={{ duration: 1.6, repeat: Infinity, repeatDelay: 2.4 }}
            role="img" aria-label="love"
          ><Heart size={24} strokeWidth={1.5} fill="currentColor" aria-hidden="true" /></motion.span> by <strong>Saurabh</strong>
        </h2>
        <p>Designed and developed with curiosity, passion, and a love for technology.</p>
        <nav className="creator-socials" aria-label="Saurabh’s social profiles">
          {socialLinks.map(({ label, href, icon }) => (
            <a key={label} href={href} target="_blank" rel="noopener noreferrer" aria-label={`${label} (opens in a new tab)`}>
              <Icon iconNode={icon} size={18} strokeWidth={1.6} aria-hidden="true" /><span>{label}</span>
            </a>
          ))}
        </nav>
      </section>
  );
}
