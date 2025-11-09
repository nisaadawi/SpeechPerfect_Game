import React from "react";
import { useNavigate } from "react-router-dom";
import { useGameSetup } from "../context/GameSetupContext";
import {
  GameSetupPage,
  OptionList,
  OptionButton,
  BackLink,
} from "../components/GameSetupPage";

const settings = [
  { label: "Study Group", value: "study_group" },
  { label: "Meeting", value: "meeting" },
  { label: "Audience", value: "audience" },
];

function SettingSelection() {
  const navigate = useNavigate();
  const { selections, setSetting } = useGameSetup();

  React.useEffect(() => {
    if (!selections.topic) {
      navigate("/game/topic", { replace: true });
    }
  }, [selections.topic, navigate]);

  const handleSelect = (value) => {
    setSetting(value);
    navigate("/game/heckler");
  };

  return (
    <GameSetupPage
      title="Select Your Setting"
      subtitle="Match the environment to how you want to practice."
      footer={
        <BackLink label="Back" onClick={() => navigate("/game/topic")} />
      }
    >
      <OptionList>
        {settings.map((option) => (
          <OptionButton
            key={option.value}
            active={selections.setting === option.value}
            onClick={() => handleSelect(option.value)}
          >
            {option.label}
          </OptionButton>
        ))}
      </OptionList>
    </GameSetupPage>
  );
}

export default SettingSelection;

