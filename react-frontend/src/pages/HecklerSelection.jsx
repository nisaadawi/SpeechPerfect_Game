import React from "react";
import { useNavigate } from "react-router-dom";
import { useGameSetup } from "../context/GameSetupContext";
import {
  GameSetupPage,
  OptionList,
  OptionButton,
  BackLink,
} from "../components/GameSetupPage";

const hecklers = [
  { label: "Easy", value: "easy" },
  { label: "Medium", value: "medium" },
  { label: "Hard", value: "hard" },
];

function HecklerSelection() {
  const navigate = useNavigate();
  const { selections, setHeckler } = useGameSetup();

  React.useEffect(() => {
    if (!selections.topic || !selections.setting) {
      navigate("/game/setting", { replace: true });
    }
  }, [selections.topic, selections.setting, navigate]);

  const handleSelect = (value) => {
    setHeckler(value);
    navigate("/game/start");
  };

  return (
    <GameSetupPage
      title="Choose Heckler Difficulty"
      subtitle="Decide how challenging your audience reactions should be."
      footer={
        <BackLink label="Back" onClick={() => navigate("/game/setting")} />
      }
    >
      <OptionList>
        {hecklers.map((option) => (
          <OptionButton
            key={option.value}
            active={selections.heckler === option.value}
            onClick={() => handleSelect(option.value)}
          >
            {option.label}
          </OptionButton>
        ))}
      </OptionList>
    </GameSetupPage>
  );
}

export default HecklerSelection;

