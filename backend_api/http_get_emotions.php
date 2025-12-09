<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");
require 'db_config.php'; // uses PDO config

try {
    $stmt = $pdo->query("SELECT emotions, timestamp FROM emotions ORDER BY reading_id DESC LIMIT 1");
    $row = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($row) {
        echo json_encode(["status" => "success", "emotion" => $row['emotions'], "timestamp" => $row['timestamp']]);
    } else {
        echo json_encode(["status" => "empty", "emotion" => null]);
    }
} catch (PDOException $e) {
    echo json_encode(["status" => "error", "message" => $e->getMessage()]);
}
?>
