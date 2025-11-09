import React from "react";
import { useNavigate } from "react-router-dom";
import { useGameSetup } from "../context/GameSetupContext";
import {
  GameSetupPage,
  BackLink,
} from "../components/GameSetupPage";

function StartGame() {
  const navigate = useNavigate();
  const { selections, reset } = useGameSetup();

  React.useEffect(() => {
    if (!selections.topic || !selections.setting || !selections.heckler) {
      navigate("/game/topic", { replace: true });
    }
  }, [navigate, selections]);

  const summaryItems = [
    { label: "Topic", value: formatLabel(selections.topic) },
    { label: "Setting", value: formatLabel(selections.setting) },
    { label: "Heckler", value: formatLabel(selections.heckler) },
  ];

  const handleStart = () => {
    navigate("/");
  };

  const handleRestart = () => {
    reset();
    navigate("/game/topic", { replace: true });
  };

  return (
    <GameSetupPage
      title="Ready to Start?"
      subtitle="Review your selections before launching the simulation."
      footer={
        <>
          <BackLink label="Adjust" onClick={() => navigate("/game/heckler")} />
          <BackLink label="Start Over" onClick={handleRestart} />
        </>
      }
    >
      <div style={summaryStyles.list}>
        {summaryItems.map((item) => (
          <div key={item.label} style={summaryStyles.item}>
            <span style={summaryStyles.label}>{item.label}</span>
            <span style={summaryStyles.value}>{item.value}</span>
          </div>
        ))}
      </div>
      <button type="button" onClick={handleStart} style={startButtonStyle}>
        Start Game
      </button>
    </GameSetupPage>
  );
}

function formatLabel(value) {
  if (!value) return "—";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const summaryStyles = {
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "1.25rem",
    textAlign: "left",
  },
  item: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "1rem 1.25rem",
    borderRadius: "12px",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
  },
  label: {
    fontSize: "1rem",
    letterSpacing: "0.05em",
    opacity: 0.7,
    textTransform: "uppercase",
  },
  value: {
    fontSize: "1.1rem",
    fontWeight: 600,
  },
};

const startButtonStyle = {
  marginTop: "1.5rem",
  padding: "1rem 1.5rem",
  borderRadius: "14px",
  border: "none",
  fontSize: "1.15rem",
  fontWeight: 700,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  cursor: "pointer",
  color: "#0b0b16",
  backgroundColor: "#55f991",
  boxShadow: "0 20px 40px rgba(85, 249, 145, 0.35)",
};

export default StartGame;

