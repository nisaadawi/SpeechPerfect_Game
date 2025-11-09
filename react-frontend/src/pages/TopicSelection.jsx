import React from "react";
import { useNavigate } from "react-router-dom";
import { useGameSetup } from "../context/GameSetupContext";
import {
  GameSetupPage,
  OptionList,
  OptionButton,
  BackLink,
} from "../components/GameSetupPage";

const topics = [
  { label: "Academic", value: "academic" },
  { label: "Technology", value: "technology" },
  { label: "Free Topic", value: "free" },
];

function TopicSelection() {
  const navigate = useNavigate();
  const { selections, setTopic, reset } = useGameSetup();

  React.useEffect(() => {
    reset();
  }, [reset]);

  const handleSelect = (value) => {
    setTopic(value);
    navigate("/game/setting");
  };

  return (
    <GameSetupPage
      title="Choose a Topic"
      subtitle="Pick the focus area for your practice session."
      footer={<BackLink label="Back to Menu" onClick={() => navigate("/main")} />}
    >
      <OptionList>
        {topics.map((option) => (
          <OptionButton
            key={option.value}
            active={selections.topic === option.value}
            onClick={() => handleSelect(option.value)}
          >
            {option.label}
          </OptionButton>
        ))}
      </OptionList>
    </GameSetupPage>
  );
}

export default TopicSelection;

