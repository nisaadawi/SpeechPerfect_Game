<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

require 'db_config.php';

$data = json_decode(file_get_contents("php://input"));

if (!$data) {
    http_response_code(400);
    echo json_encode(["success" => false, "message" => "Invalid JSON data."]);
    exit();
}

if (
    isset($data->username) && isset($data->age) && isset($data->gender) &&
    isset($data->email) && isset($data->password)
) {
    $username = htmlspecialchars($data->username);
    $age = (int)$data->age;
    $gender = htmlspecialchars($data->gender);
    $email = strtolower(trim($data->email));
    $password = password_hash($data->password, PASSWORD_DEFAULT);

    // Check if email already exists
    $stmt = $pdo->prepare("SELECT id FROM users WHERE email = ?");
    $stmt->execute([$email]);

    if ($stmt->rowCount() > 0) {
        echo json_encode(["success" => false, "message" => "Email already registered."]);
        exit();
    }

    // Insert user
    $stmt = $pdo->prepare("INSERT INTO users (username, age, gender, email, password) VALUES (?, ?, ?, ?, ?)");
    $stmt->execute([$username, $age, $gender, $email, $password]);

    echo json_encode(["success" => true, "message" => "Registration successful."]);
} else {
    echo json_encode(["success" => false, "message" => "Invalid input."]);
}
?>
