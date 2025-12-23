<?php
/**
 * Update session with all analysis data
 * Called after analysis is complete
 */
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

require_once 'db_config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["success" => false, "message" => "Only POST method allowed"]);
    exit();
}

$input = json_decode(file_get_contents('php://input'), true);
$session_id = intval($input['session_id'] ?? 0);

if (!$session_id) {
    echo json_encode(["success" => false, "message" => "Missing session_id"]);
    exit();
}

try {
    // Update session with all data
    $stmt = $pdo->prepare("
        UPDATE tbl_session SET
            attention_percentage = ?,
            attention_cat_label = ?,
            audio_file = ?,
            filler_per_min = ?,
            filler_cat_label = ?,
            avg_hr = ?,
            hr_cat_label = ?,
            mfcc_std_dev = ?,
            mfcc_cat_label = ?,
            avg_pause = ?,
            pause_cat_label = ?,
            stress_probability = ?,
            stress_cat_label = ?,
            wpm = ?,
            wpm_cat_label = ?,
            speech_param = ?,
            speech_score = ?,
            speech_label = ?
        WHERE session_id = ?
    ");
    
    $stmt->execute([
        $input['attention_percentage'] ?? null,
        $input['attention_cat_label'] ?? null,
        $input['audio_file'] ?? null,
        $input['filler_per_min'] ?? null,
        $input['filler_cat_label'] ?? null,
        $input['avg_hr'] ?? null,
        $input['hr_cat_label'] ?? null,
        $input['mfcc_std_dev'] ?? null,
        $input['mfcc_cat_label'] ?? null,
        $input['avg_pause'] ?? null,
        $input['pause_cat_label'] ?? null,
        $input['stress_probability'] ?? null,
        $input['stress_cat_label'] ?? null,
        $input['wpm'] ?? null,
        $input['wpm_cat_label'] ?? null,
        $input['speech_param'] ?? null,
        $input['speech_score'] ?? null,
        $input['speech_label'] ?? null,
        $session_id
    ]);
    
    echo json_encode([
        "success" => true,
        "message" => "Session updated successfully"
    ]);
    
} catch (PDOException $e) {
    echo json_encode([
        "success" => false,
        "message" => "Database error: " . $e->getMessage()
    ]);
}
?>

