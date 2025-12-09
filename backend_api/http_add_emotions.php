<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");
require 'db_config.php'; // Uses your PDO config

$emotion = $_POST['emotion'] ?? '';

if ($emotion !== '') {
    try {
        $stmt = $pdo->prepare("INSERT INTO emotions (emotions) VALUES (:emotion)");
        $stmt->execute(['emotion' => $emotion]);

        if ($stmt->rowCount() > 0) {
            echo json_encode(["status" => "success", "message" => "Emotion stored"]);
        } else {
            echo json_encode(["status" => "fail", "message" => "Insert failed"]);
        }
    } catch (PDOException $e) {
        echo json_encode(["status" => "error", "message" => "DB error: " . $e->getMessage()]);
    }
} else {
    echo json_encode(["status" => "error", "message" => "No emotion received"]);
}
?>
