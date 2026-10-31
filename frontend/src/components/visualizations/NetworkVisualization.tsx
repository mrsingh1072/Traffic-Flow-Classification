import { motion, useReducedMotion } from "motion/react";

export function NetworkVisualization() {
  const reduced = useReducedMotion();
  const reveal = {
    initial: { pathLength: reduced ? 1 : 0 },
    animate: { pathLength: 1 },
    transition: { duration: reduced ? 0 : 0.65, ease: "easeOut" as const },
  };
  return (
    <figure className="topology-panel">
      <figcaption className="topology-heading">
        <div>
          <h2>5G network topology</h2>
        </div>
        <span className="diagram-caption">Illustrative architecture</span>
      </figcaption>
      <div
        className="topology-scroll"
        tabIndex={0}
        aria-label="5G architecture diagram; scroll horizontally on small screens"
      >
        <svg
          className="network-diagram"
          viewBox="0 0 1120 230"
          role="img"
          aria-labelledby="topology-title topology-desc"
        >
          <title id="topology-title">
            UE to gNodeB to 5G Core, through the UPF, to a destination data
            network
          </title>
          <desc id="topology-desc">
            Solid arrows show the user-plane path. The UPF is inside the 5G core
            boundary. Dashed N2 and N4 connections represent control interfaces
            to AMF and SMF. This diagram illustrates network architecture and is
            not live monitoring.
          </desc>
          <defs>
            <marker
              id="path-arrow"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path
                d="M1 1L7 4L1 7"
                fill="none"
                stroke="#686A65"
                strokeWidth="1.2"
              />
            </marker>
            <pattern
              id="diagram-dots"
              width="20"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r=".65" fill="#464741" />
            </pattern>
          </defs>

          <rect
            x="506"
            y="14"
            width="343"
            height="201"
            rx="3"
            fill="#F5F5F2"
            stroke="#D4D5CC"
            strokeDasharray="4 5"
          />
          <text x="527" y="39" className="svg-zone">
            5G CORE
          </text>
          <g className="control-node">
            <rect x="553" y="57" width="96" height="42" rx="2" />
            <text x="601" y="83">
              AMF
            </text>
            <rect x="704" y="57" width="96" height="42" rx="2" />
            <text x="752" y="83">
              SMF
            </text>
          </g>
          <motion.path
            initial={{ opacity: reduced ? 1 : 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            d="M372 126V78H546"
            className="control-line"
            markerEnd="url(#path-arrow)"
          />
          <text x="436" y="68" className="svg-interface">
            N2
          </text>
          <path d="M649 78H704" className="control-line" />
          <text x="676" y="68" className="svg-interface">
            N11
          </text>
          <motion.path
            initial={{ opacity: reduced ? 1 : 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            d="M752 99V132"
            className="control-line"
            markerEnd="url(#path-arrow)"
          />
          <text x="771" y="119" className="svg-interface">
            N4
          </text>
          <motion.path
            {...reveal}
            d="M163 158H304"
            className="user-line"
            markerEnd="url(#path-arrow)"
          />
          <text x="232" y="144" className="svg-interface">
            RADIO ACCESS
          </text>
          <motion.path
            {...reveal}
            d="M424 158H659"
            className="user-line"
            markerEnd="url(#path-arrow)"
          />
          <text x="540" y="144" className="svg-interface">
            N3
          </text>
          <motion.path
            {...reveal}
            d="M800 158H956"
            className="user-line"
            markerEnd="url(#path-arrow)"
          />
          <text x="878" y="144" className="svg-interface">
            N6
          </text>
          <g className="diagram-device" transform="translate(55 126)">
            <rect width="108" height="64" rx="3" />
            <g
              transform="translate(14 16)"
              fill="none"
              stroke="#686A65"
              strokeWidth="1.6"
            >
              <rect x="1" width="17" height="28" rx="3" />
              <path d="M7 23H12" />
            </g>
            <text x="67" y="37" className="svg-node">
              UE
            </text>
          </g>
          <text x="109" y="213" className="svg-description">
            User equipment
          </text>
          <g className="diagram-device" transform="translate(305 126)">
            <rect width="119" height="64" rx="3" />
            <g
              transform="translate(15 17)"
              fill="none"
              stroke="#686A65"
              strokeWidth="1.5"
            >
              <path d="M12 0V29M4 29H20M3 7Q12 -2 21 7M-2 1Q12 -11 26 1M5 18L19 29M19 18L5 29" />
            </g>
            <text x="78" y="37" className="svg-node">
              gNodeB
            </text>
          </g>
          <text x="364" y="213" className="svg-description">
            Radio access network
          </text>
          <g className="diagram-device upf-node" transform="translate(660 126)">
            <rect width="140" height="64" rx="3" />
            <g
              transform="translate(16 20)"
              fill="none"
              stroke="#686A65"
              strokeWidth="1.5"
            >
              <rect width="31" height="24" rx="2" />
              <path d="M5 7H25M5 16H9M14 16H18M23 16H26" />
            </g>
            <text x="96" y="37" className="svg-node">
              UPF
            </text>
          </g>
          <text x="730" y="210" className="svg-description">
            User Plane Function
          </text>
          <g className="diagram-device" transform="translate(957 126)">
            <rect width="133" height="64" rx="3" />
            <g
              transform="translate(13 18)"
              fill="none"
              stroke="#686A65"
              strokeWidth="1.5"
            >
              <rect width="27" height="12" rx="2" />
              <rect y="16" width="27" height="12" rx="2" />
              <path d="M5 6H8M5 22H8" />
            </g>
            <text x="88" y="37" className="svg-node">
              DN
            </text>
          </g>
          <text x="1023" y="213" className="svg-description">
            Destination network
          </text>
        </svg>
      </div>
      <MobileTopology />
      <div className="topology-foot">
        <div className="diagram-legend">
          <span>
            <i />
            User-plane path
          </span>
          <span>
            <i className="dashed" />
            Control interface
          </span>
        </div>
        <p>
          Illustrative context only. TrafficFlow analyzes supplied feature
          records.
        </p>
      </div>
    </figure>
  );
}

function MobileTopology() {
  return (
    <svg
      className="mobile-topology"
      viewBox="0 0 330 475"
      role="img"
      aria-labelledby="mobile-topology-title mobile-topology-desc"
    >
      <title id="mobile-topology-title">
        5G path: UE, gNodeB, core UPF, destination
      </title>
      <desc id="mobile-topology-desc">
        The vertical solid path shows user-plane traffic. The UPF is inside the
        5G core. Dashed N2, N11, and N4 interfaces connect the control functions
        AMF and SMF.
      </desc>
      <defs>
        <marker
          id="mobile-arrow"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path
            d="M1 1L7 4L1 7"
            fill="none"
            stroke="#686A65"
            strokeWidth="1.2"
          />
        </marker>
      </defs>
      <rect
        x="24"
        y="195"
        width="288"
        height="163"
        rx="3"
        fill="#F5F5F2"
        stroke="#D4D5CC"
        strokeDasharray="4 5"
      />
      <text x="39" y="219" className="svg-zone">
        5G CORE
      </text>
      <path
        d="M114 60V107M114 158V277M114 328V404"
        className="user-line"
        markerEnd="url(#mobile-arrow)"
      />
      <text x="171" y="88" className="svg-interface">
        Radio access
      </text>
      <text x="135" y="254" className="svg-interface">
        N3
      </text>
      <text x="135" y="387" className="svg-interface">
        N6
      </text>
      <path
        d="M182 132H256V200"
        className="control-line"
        markerEnd="url(#mobile-arrow)"
      />
      <text x="278" y="173" className="svg-interface">
        N2
      </text>
      <path d="M256 243V271" className="control-line" />
      <text x="285" y="264" className="svg-interface">
        N11
      </text>
      <path
        d="M218 293H199V303H183"
        className="control-line"
        markerEnd="url(#mobile-arrow)"
      />
      <text x="196" y="280" className="svg-interface">
        N4
      </text>
      <g className="control-node">
        <rect x="218" y="201" width="76" height="42" rx="2" />
        <text x="256" y="228">
          AMF
        </text>
        <rect x="218" y="272" width="76" height="42" rx="2" />
        <text x="256" y="299">
          SMF
        </text>
      </g>
      {[
        { y: 10, label: "UE", description: "User equipment" },
        { y: 108, label: "gNodeB", description: "Radio access network" },
        { y: 278, label: "UPF", description: "User Plane Function" },
        { y: 405, label: "Destination", description: "Data network" },
      ].map((node) => (
        <g
          key={node.label}
          className={
            node.label === "UPF" ? "diagram-device upf-node" : "diagram-device"
          }
          transform={`translate(46 ${node.y})`}
        >
          <rect width="136" height="50" rx="3" />
          <text x="68" y="22" className="svg-node">
            {node.label}
          </text>
          <text x="68" y="40" className="mobile-node-description">
            {node.description}
          </text>
        </g>
      ))}
    </svg>
  );
}
