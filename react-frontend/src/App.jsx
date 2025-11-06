import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import MainMenu from "./pages/MainMenu";
import Dashboard from "./pages/Dashboard";
import Help from "./pages/Help";

function Home() {
  const [focused, setFocused] = useState(null);
  const videoUrl = "http://localhost:5000/video_feed";

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:5000/focus")
        .then((res) => res.json())
        .then((data) => setFocused(data.focused))
        .catch((err) => console.error(err));
    }, 1000); // every second

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        alignItems: "center",
        justifyItems: "center",
        justifyContent: "center",
        gap: "2rem",
        flexDirection: "column",
        backgroundColor: "#111",
        color: "white",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "2rem",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            backgroundColor: "#1e1e1e",
            borderRadius: "12px",
            padding: "1.5rem",
            minWidth: "260px",
            textAlign: "center",
            boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
          }}
        >
          <div style={{ fontSize: "1.2rem", opacity: 0.8 }}>Focus Status</div>
          <div
            style={{
              marginTop: "1rem",
              fontSize: "2rem",
              color:
                focused === null ? "#ffeb3b" : focused ? "#4caf50" : "#f44336",
            }}
          >
            {focused === null
              ? "Loading..."
              : focused
              ? "User is Focused 👀"
              : "User is Not Focused 😴"}
          </div>
        </div>

        <div
          style={{
            position: "relative",
            width: "480px",
            maxWidth: "90vw",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 10px 25px rgba(0,0,0,0.4)",
            backgroundColor: "#000",
          }}
        >
          <img
            src={videoUrl}
            alt="Gaze camera feed"
            style={{ width: "100%", display: "block" }}
          />
        </div>
      </div>
    </div>
  );
}

// Protected Route Component
function ProtectedRoute({ children }) {
  const user = localStorage.getItem("user");
  return user ? children : <Navigate to="/login" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/main"
          element={
            <ProtectedRoute>
              <MainMenu />
            </ProtectedRoute>
          }
        />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/help"
          element={
            <ProtectedRoute>
              <Help />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/main" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
