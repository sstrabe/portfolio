// The single source of truth for portfolio content. Both the plain page
// (prerendered into index.html at build time) and the immersive stations
// are generated from this file.
//
// Station 0 is the "home" station (about + contact); each project becomes
// one more station on its own orbit. The renderer has room for 8 stations.
//
// TODO(owner): replace the placeholder entries below with real content.

export interface Link {
  label: string;
  href: string;
}

export interface Project {
  id: string;
  title: string;
  tagline: string;
  year?: string;
  role?: string;
  /** Paragraphs of plain text. */
  description: string[];
  tags: string[];
  links: Link[];
  /** Accent colour for the station's card. */
  accent: string;
}

export interface Portfolio {
  owner: {
    name: string;
    headline: string;
    bio: string[];
    links: Link[];
  };
  projects: Project[];
}

export const MAX_STATIONS = 8;

export const portfolio: Portfolio = {
  owner: {
    name: "Your Name",
    headline: "Software engineer — replace this headline",
    bio: [
      "Placeholder bio. Say who you are, what you build and what you are looking for.",
      "This page doubles as the fast, plain version of an interactive portfolio set around a spinning black hole.",
    ],
    links: [
      { label: "GitHub", href: "https://github.com/" },
      { label: "Email", href: "mailto:you@example.com" },
    ],
  },
  projects: [
    {
      id: "kerr-nucleus",
      title: "Kerr Nucleus",
      tagline: "This portfolio: a star cluster around a spinning black hole, rendered from your past light cone.",
      year: "2026",
      role: "Design, physics, engineering",
      description: [
        "Every star follows an exact Kerr geodesic in Kerr–Schild coordinates. Stars nudge each other through weak, retarded, velocity-extrapolated pulls that respect light-speed causality, and tight orbits of compact objects decay through 2.5PN radiation reaction.",
        "You pilot a ship whose worldline can never reach c. The simulation clock is your proper time, so high speeds and deep dives make the universe fast-forward. The view is ray traced along null geodesics: lensing, aberration, Doppler colour shifts and beaming all come out of the same equations.",
        "Physics and rendering are Rust compiled to WebAssembly on WebGPU; the content is plain TypeScript and HTML, with this page as the fallback.",
      ],
      tags: ["Rust", "WebAssembly", "WebGPU", "WGSL", "General relativity", "TypeScript"],
      links: [{ label: "Source", href: "https://github.com/sstrabe/portfolio" }],
      accent: "#7fd6ff",
    },
    {
      id: "project-two",
      title: "Project Two",
      tagline: "Placeholder: one sentence on what it does and why it matters.",
      year: "2025",
      role: "Your role",
      description: [
        "Placeholder description. Explain the problem, your approach and the outcome in a few short paragraphs.",
      ],
      tags: ["Tag", "Another tag"],
      links: [{ label: "Link", href: "https://example.com/" }],
      accent: "#ffb86b",
    },
    {
      id: "project-three",
      title: "Project Three",
      tagline: "Placeholder: one sentence on what it does and why it matters.",
      year: "2024",
      role: "Your role",
      description: ["Placeholder description."],
      tags: ["Tag"],
      links: [],
      accent: "#b59cff",
    },
    {
      id: "project-four",
      title: "Project Four",
      tagline: "Placeholder: one sentence on what it does and why it matters.",
      year: "2023",
      role: "Your role",
      description: ["Placeholder description."],
      tags: ["Tag"],
      links: [],
      accent: "#8cf5a8",
    },
    {
      id: "project-five",
      title: "Project Five",
      tagline: "Placeholder: one sentence on what it does and why it matters.",
      year: "2022",
      role: "Your role",
      description: ["Placeholder description."],
      tags: ["Tag"],
      links: [],
      accent: "#ff8fb3",
    },
  ],
};

/** A station in the immersive world: home first, then one per project. */
export interface Station {
  index: number;
  id: string;
  title: string;
  subtitle: string;
  accent: string;
  project?: Project;
}

export function stations(p: Portfolio = portfolio): Station[] {
  const home: Station = {
    index: 0,
    id: "about",
    title: p.owner.name,
    subtitle: p.owner.headline,
    accent: "#e9edf5",
  };
  const rest = p.projects.slice(0, MAX_STATIONS - 1).map((project, i) => ({
    index: i + 1,
    id: project.id,
    title: project.title,
    subtitle: project.tagline,
    accent: project.accent,
    project,
  }));
  return [home, ...rest];
}
