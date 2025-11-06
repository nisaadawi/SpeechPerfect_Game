import React from "react";

function Dashboard() {
  let user = null;
  try {
    user = JSON.parse(localStorage.getItem("user") || "null");
  } catch (error) {
    user = null;
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#101322",
        color: "#f0f1f5",
        gap: "1rem",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <h1 style={{ fontSize: "2.4rem", letterSpacing: "0.05em" }}>Dashboard</h1>
      <p style={{ opacity: 0.75 }}>
        {user
          ? `Hi ${user.username}, your personalized dashboard will be available here soon.`
          : "Your personalized dashboard will be available here soon."}
      </p>
    </div>
  );
}

export default Dashboard;

