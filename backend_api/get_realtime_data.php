<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

require_once 'db_config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["success" => false, "message" => "Only POST method allowed"]);
    exit();
}

// Get JSON input
$input = json_decode(file_get_contents('php://input'), true);

if (!$input) {
    echo json_encode(["success" => false, "message" => "Invalid JSON input"]);
    exit();
}

// Validate required fields
if (!isset($input['user_id'])) {
    echo json_encode(["success" => false, "message" => "Missing required field: user_id"]);
    exit();
}

$user_id = intval($input['user_id']);

try {
    // Fetch latest heart rate data for user
    $hr_stmt = $pdo->prepare("
        SELECT hr_min, hr_max, hr_avg, hr_status 
        FROM tbl_heart_rate 
        WHERE id = ? 
        ORDER BY hr_id DESC 
        LIMIT 1
    ");
    $hr_stmt->execute([$user_id]);
    $heart_rate_data = $hr_stmt->fetch(PDO::FETCH_ASSOC);
    
    // Fetch latest attention data for user
    $att_stmt = $pdo->prepare("
        SELECT focus_percentage, attention_status 
        FROM tbl_attention 
        WHERE id = ? 
        ORDER BY attention_id DESC 
        LIMIT 1
    ");
    $att_stmt->execute([$user_id]);
    $attention_data = $att_stmt->fetch(PDO::FETCH_ASSOC);
    
    echo json_encode([
        "success" => true,
        "heart_rate" => $heart_rate_data ? [
            "hr_min" => intval($heart_rate_data['hr_min']),
            "hr_max" => intval($heart_rate_data['hr_max']),
            "hr_avg" => intval($heart_rate_data['hr_avg']),
            "hr_status" => $heart_rate_data['hr_status']
        ] : null,
        "attention" => $attention_data ? [
            "focus_percentage" => intval($attention_data['focus_percentage']),
            "attention_status" => $attention_data['attention_status']
        ] : null
    ]);
    
} catch (PDOException $e) {
    echo json_encode([
        "success" => false,
        "message" => "Database error: " . $e->getMessage()
    ]);
} catch (Exception $e) {
    echo json_encode([
        "success" => false,
        "message" => "Error: " . $e->getMessage()
    ]);
}
?>

