import React from "react";

export function GameSetupPage({ title, subtitle, children, footer }) {
  return (
    <div style={pageStyles.wrapper}>
      <div style={pageStyles.card}>
        <h1 style={pageStyles.title}>{title}</h1>
        {subtitle ? <p style={pageStyles.subtitle}>{subtitle}</p> : null}
        {children}
        {footer ? <div style={pageStyles.footer}>{footer}</div> : null}
      </div>
    </div>
  );
}

export function OptionList({ children }) {
  return <div style={pageStyles.buttonGrid}>{children}</div>;
}

export function OptionButton({ active, onClick, children }) {
  return (
    <button type="button" onClick={onClick} style={buttonStyle({ active })}>
      {children}
    </button>
  );
}

export const BackLink = ({ label = "Back", onClick }) => (
  <button type="button" onClick={onClick} style={backButtonStyle}>
    {label}
  </button>
);

const pageStyles = {
  wrapper: {
    minHeight: "100vh",
    width: "100vw",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "radial-gradient(circle at top, #1f2353 0%, #0c0f24 80%)",
    color: "#fefefe",
    padding: "2rem",
    boxSizing: "border-box",
  },
  card: {
    width: "100%",
    maxWidth: "560px",
    backgroundColor: "rgba(12, 13, 24, 0.88)",
    borderRadius: "20px",
    padding: "3rem 2.5rem",
    boxShadow: "0 28px 60px rgba(0, 0, 0, 0.45)",
    border: "1px solid rgba(255, 255, 255, 0.05)",
    textAlign: "center",
    display: "flex",
    flexDirection: "column",
    gap: "2.5rem",
  },
  title: {
    fontSize: "2.4rem",
    marginBottom: "0.25rem",
    letterSpacing: "0.06em",
    textTransform: "uppercase",
  },
  subtitle: {
    fontSize: "1.05rem",
    opacity: 0.8,
  },
  buttonGrid: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  footer: {
    display: "flex",
    justifyContent: "center",
    gap: "1.5rem",
    flexWrap: "wrap",
  },
};

function buttonStyle({ active = false }) {
  return {
    padding: "1rem 1.25rem",
    borderRadius: "12px",
    border: "none",
    fontSize: "1.1rem",
    fontWeight: 600,
    letterSpacing: "0.04em",
    cursor: "pointer",
    color: active ? "#0b0b16" : "#fefefe",
    backgroundColor: active ? "#ffde59" : "rgba(255, 255, 255, 0.08)",
    transition: "transform 0.2s ease, box-shadow 0.3s ease, background-color 0.3s ease",
    boxShadow: active
      ? "0 22px 32px rgba(255, 222, 89, 0.35)"
      : "0 18px 28px rgba(0, 0, 0, 0.25)",
    textTransform: "uppercase",
  };
}

const backButtonStyle = {
  padding: "0.75rem 1.25rem",
  borderRadius: "999px",
  border: "1px solid rgba(255,255,255,0.2)",
  backgroundColor: "transparent",
  color: "#fefefe",
  fontSize: "0.95rem",
  letterSpacing: "0.05em",
  cursor: "pointer",
  transition: "background-color 0.2s ease",
};

export default GameSetupPage;

