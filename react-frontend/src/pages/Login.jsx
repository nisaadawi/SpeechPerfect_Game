import React, { useState, useEffect } from "react";
import { useNavigate, Link, useLocation } from "react-router-dom";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (location.state?.message) {
      setSuccess(location.state.message);
      // Clear the state to prevent showing message on refresh
      window.history.replaceState({}, document.title);
    }
  }, [location]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Update this URL to match your PHP backend endpoint
      const response = await fetch("https://humancc.site/ndhos/api_backend/login.php", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success) {
        // Store user data in localStorage
        localStorage.setItem("user", JSON.stringify(data.user));
        // Navigate to main menu
        navigate("/main");
      } else {
        setError(data.message || "Login failed. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection.");
      console.error("Login error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#1a1a1a",
        padding: "2rem",
      }}
    >
      <div
        style={{
          backgroundColor: "#2a2a2a",
          borderRadius: "16px",
          padding: "3rem",
          width: "100%",
          maxWidth: "400px",
          boxShadow: "0 10px 40px rgba(0,0,0,0.5)",
        }}
      >
        <h1
          style={{
            color: "#fff",
            marginBottom: "0.5rem",
            fontSize: "2rem",
            textAlign: "center",
          }}
        >
          Login
        </h1>
        <p
          style={{
            color: "#aaa",
            marginBottom: "2rem",
            textAlign: "center",
            fontSize: "0.9rem",
          }}
        >
          Welcome back! Please sign in to continue.
        </p>

        {success && (
          <div
            style={{
              backgroundColor: "#4caf50",
              color: "#fff",
              padding: "0.75rem",
              borderRadius: "8px",
              marginBottom: "1.5rem",
              fontSize: "0.9rem",
            }}
          >
            {success}
          </div>
        )}

        {error && (
          <div
            style={{
              backgroundColor: "#ff4444",
              color: "#fff",
              padding: "0.75rem",
              borderRadius: "8px",
              marginBottom: "1.5rem",
              fontSize: "0.9rem",
            }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1.5rem" }}>
            <label
              style={{
                display: "block",
                color: "#fff",
                marginBottom: "0.5rem",
                fontSize: "0.9rem",
                fontWeight: "500",
              }}
            >
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{
                width: "100%",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "1px solid #444",
                backgroundColor: "#1a1a1a",
                color: "#fff",
                fontSize: "1rem",
                boxSizing: "border-box",
              }}
              placeholder="Enter your email"
            />
          </div>

          <div style={{ marginBottom: "2rem" }}>
            <label
              style={{
                display: "block",
                color: "#fff",
                marginBottom: "0.5rem",
                fontSize: "0.9rem",
                fontWeight: "500",
              }}
            >
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{
                width: "100%",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "1px solid #444",
                backgroundColor: "#1a1a1a",
                color: "#fff",
                fontSize: "1rem",
                boxSizing: "border-box",
              }}
              placeholder="Enter your password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.875rem",
              backgroundColor: loading ? "#666" : "#4caf50",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              fontSize: "1rem",
              fontWeight: "600",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "background-color 0.3s",
              marginBottom: "1.5rem",
            }}
            onMouseOver={(e) => {
              if (!loading) e.target.style.backgroundColor = "#45a049";
            }}
            onMouseOut={(e) => {
              if (!loading) e.target.style.backgroundColor = "#4caf50";
            }}
          >
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>

        <p
          style={{
            color: "#aaa",
            textAlign: "center",
            fontSize: "0.9rem",
          }}
        >
          Don't have an account?{" "}
          <Link
            to="/register"
            style={{
              color: "#4caf50",
              textDecoration: "none",
              fontWeight: "600",
            }}
          >
            Register here
          </Link>
        </p>
      </div>
    </div>
  );
}

export default Login;

