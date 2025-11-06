import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

function Register() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
    age: "",
    gender: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (!formData.age || parseInt(formData.age) < 1) {
      setError("Please enter a valid age.");
      return;
    }

    if (!formData.gender) {
      setError("Please select a gender.");
      return;
    }

    setLoading(true);

    try {
      // Update this URL to match your PHP backend endpoint
      const response = await fetch("https://humancc.site/ndhos/api_backend/register.php", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password,
          age: parseInt(formData.age),
          gender: formData.gender,
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Navigate to login page after successful registration
        navigate("/login", { state: { message: "Registration successful! Please login." } });
      } else {
        setError(data.message || "Registration failed. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection.");
      console.error("Registration error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        overflowY: "auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#1a1a1a",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          backgroundColor: "#2a2a2a",
          borderRadius: "16px",
          padding: "3rem",
          width: "100%",
          maxWidth: "500px",
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
          Register
        </h1>
        <p
          style={{
            color: "#aaa",
            marginBottom: "2rem",
            textAlign: "center",
            fontSize: "0.9rem",
          }}
        >
          Create a new account to get started.
        </p>

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
              Username
            </label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
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
              placeholder="Enter your username"
            />
          </div>

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
              name="email"
              value={formData.email}
              onChange={handleChange}
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

          <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
            <div style={{ flex: 1 }}>
              <label
                style={{
                  display: "block",
                  color: "#fff",
                  marginBottom: "0.5rem",
                  fontSize: "0.9rem",
                  fontWeight: "500",
                }}
              >
                Age
              </label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                required
                min="1"
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
                placeholder="Age"
              />
            </div>

            <div style={{ flex: 1 }}>
              <label
                style={{
                  display: "block",
                  color: "#fff",
                  marginBottom: "0.5rem",
                  fontSize: "0.9rem",
                  fontWeight: "500",
                }}
              >
                Gender
              </label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
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
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

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
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
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
              Confirm Password
            </label>
            <input
              type="password"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
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
              placeholder="Confirm your password"
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
            {loading ? "Registering..." : "Register"}
          </button>
        </form>

        <p
          style={{
            color: "#aaa",
            textAlign: "center",
            fontSize: "0.9rem",
          }}
        >
          Already have an account?{" "}
          <Link
            to="/login"
            style={{
              color: "#4caf50",
              textDecoration: "none",
              fontWeight: "600",
            }}
          >
            Login here
          </Link>
        </p>
      </div>
    </div>
  );
}

export default Register;

