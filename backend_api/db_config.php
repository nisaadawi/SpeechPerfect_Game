<?php
$host = "localhost";       // Your DB host
$dbname = "humancmt_nd_speechperfect"; // Your DB name
$user = "humancmt_nd_admin";   // Your DB username
$pass = "Wereenz0909?";   // Your DB password

try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8mb4", $user, $pass);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    echo json_encode(["success" => false, "message" => "Database connection failed: " . $e->getMessage()]);
    exit();
}
?>
