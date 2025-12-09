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

if (isset($data->email) && isset($data->password)) {
    $email = strtolower(trim($data->email));
    $password = $data->password;

    $stmt = $pdo->prepare("SELECT id, username, age, gender, email, password FROM users WHERE email = ?");
    $stmt->execute([$email]);

    if ($stmt->rowCount() === 1) {
        $user = $stmt->fetch(PDO::FETCH_ASSOC);
        if (password_verify($password, $user['password'])) {
            // Remove password before sending
            unset($user['password']);
            echo json_encode(["success" => true, "user" => $user]);
            exit();
        }
    }

    echo json_encode(["success" => false, "message" => "Invalid credentials."]);
} else {
    echo json_encode(["success" => false, "message" => "Missing email or password."]);
}
?>
