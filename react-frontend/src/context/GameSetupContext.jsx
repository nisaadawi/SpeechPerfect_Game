import React, {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
} from "react";

const GameSetupContext = createContext(null);

export function GameSetupProvider({ children }) {
  const [selections, setSelections] = useState({
    topic: null,
    setting: null,
    heckler: null,
  });

  const setTopic = useCallback((topic) => {
    setSelections((prev) => ({
      ...prev,
      topic,
    }));
  }, []);

  const setSetting = useCallback((setting) => {
    setSelections((prev) => ({
      ...prev,
      setting,
    }));
  }, []);

  const setHeckler = useCallback((heckler) => {
    setSelections((prev) => ({
      ...prev,
      heckler,
    }));
  }, []);

  const reset = useCallback(() => {
    setSelections({
      topic: null,
      setting: null,
      heckler: null,
    });
  }, []);

  const value = useMemo(
    () => ({ selections, setTopic, setSetting, setHeckler, reset }),
    [selections, setTopic, setSetting, setHeckler, reset]
  );

  return (
    <GameSetupContext.Provider value={value}>
      {children}
    </GameSetupContext.Provider>
  );
}

export function useGameSetup() {
  const context = useContext(GameSetupContext);
  if (!context) {
    throw new Error("useGameSetup must be used within a GameSetupProvider");
  }
  return context;
}

