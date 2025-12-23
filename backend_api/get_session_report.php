<?php
/**
 * Get detailed session report by session_id
 * Fetches complete session data from tbl_session and related tables
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
$user_id = intval($input['user_id'] ?? 0);
$session_id = intval($input['session_id'] ?? 0);

// Support both session_id and timestamp for backward compatibility
$timestamp = $input['timestamp'] ?? null;

if (!$user_id) {
    echo json_encode(["success" => false, "message" => "Missing user_id"]);
    exit();
}

if (!$session_id && !$timestamp) {
    echo json_encode(["success" => false, "message" => "Missing session_id or timestamp"]);
    exit();
}

try {
    // Fetch session from master table
    if ($session_id) {
        $session_stmt = $pdo->prepare("
            SELECT * FROM tbl_session 
            WHERE session_id = ? AND id = ?
            LIMIT 1
        ");
        $session_stmt->execute([$session_id, $user_id]);
    } else {
        // Fallback: use timestamp
        $session_stmt = $pdo->prepare("
            SELECT * FROM tbl_session 
            WHERE id = ? AND timestamp = ?
            LIMIT 1
        ");
        $session_stmt->execute([$user_id, $timestamp]);
    }
    
    $session = $session_stmt->fetch(PDO::FETCH_ASSOC);
    
    if (!$session) {
        echo json_encode(["success" => false, "message" => "Session not found"]);
        exit();
    }
    
    // Fetch detailed data from related tables (if needed for additional details)
    $wpm_stmt = $pdo->prepare("
        SELECT transcript, duration_min, word_count, word_per_min, wpm_label
        FROM tbl_wpm 
        WHERE session_id = ? 
        LIMIT 1
    ");
    $wpm_stmt->execute([$session['session_id']]);
    $wpm_data = $wpm_stmt->fetch(PDO::FETCH_ASSOC);
    
    $filler_stmt = $pdo->prepare("
        SELECT filler_count, detect_filler, filler_per_min, filler_label
        FROM tbl_filler 
        WHERE session_id = ? 
        LIMIT 1
    ");
    $filler_stmt->execute([$session['session_id']]);
    $filler_data = $filler_stmt->fetch(PDO::FETCH_ASSOC);
    
    $pause_stmt = $pdo->prepare("
        SELECT pause_count, avg_pause_sec, pause_label
        FROM tbl_pause 
        WHERE session_id = ? 
        LIMIT 1
    ");
    $pause_stmt->execute([$session['session_id']]);
    $pause_data = $pause_stmt->fetch(PDO::FETCH_ASSOC);
    
    $hr_stmt = $pdo->prepare("
        SELECT hr_min, hr_max, hr_avg, hr_status
        FROM tbl_heart_rate 
        WHERE session_id = ? 
        LIMIT 1
    ");
    $hr_stmt->execute([$session['session_id']]);
    $hr_data = $hr_stmt->fetch(PDO::FETCH_ASSOC);
    
    $att_stmt = $pdo->prepare("
        SELECT focus_percentage, attention_status
        FROM tbl_attention 
        WHERE session_id = ? 
        LIMIT 1
    ");
    $att_stmt->execute([$session['session_id']]);
    $att_data = $att_stmt->fetch(PDO::FETCH_ASSOC);
    
    // Construct analysis data in the same format as SpeechAnalysisResults expects
    $analysis = [
        "speech_param" => floatval($session['speech_param'] ?? 0),
        "speech_score" => floatval($session['speech_score'] ?? 0),
        "transcript" => $wpm_data['transcript'] ?? "",
        "speech_rate_wpm" => floatval($session['wpm'] ?? 0),
        "wpm_label" => $session['wpm_cat_label'] >= 1.0 ? 'Good' : ($session['wpm_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
        "filler_count" => intval($filler_data['filler_count'] ?? 0),
        "fillers_per_min" => floatval($session['filler_per_min'] ?? 0),
        "filler_label" => $session['filler_cat_label'] >= 1.0 ? 'Good' : ($session['filler_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
        "avg_pause_sec" => floatval($session['avg_pause'] ?? 0),
        "pause_label" => $session['pause_cat_label'] >= 1.0 ? 'Good' : ($session['pause_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
        "mfcc_std_mean_raw" => floatval($session['mfcc_std_dev'] ?? 0),
        "mfcc_std_label" => $session['mfcc_cat_label'] >= 1.0 ? 'Good' : ($session['mfcc_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
        "stress_probability" => floatval($session['stress_probability'] ?? 0),
        "stress_label" => $session['stress_cat_label'] >= 1.0 ? 'Stressed' : 'Not Stressed',
        "duration_sec" => floatval($wpm_data['duration_min'] ?? 0) * 60,
        "word_count" => intval($wpm_data['word_count'] ?? 0),
    ];
    
    echo json_encode([
        "success" => true,
        "status" => "success",
        "analysis" => $analysis,
        "heart_rate" => $hr_data ? [
            "hr_min" => intval($hr_data['hr_min']),
            "hr_max" => intval($hr_data['hr_max']),
            "hr_avg" => floatval($session['avg_hr'] ?? $hr_data['hr_avg']),
            "hr_status" => $hr_data['hr_status']
        ] : ($session['avg_hr'] ? [
            "hr_avg" => floatval($session['avg_hr']),
            "hr_status" => $session['hr_cat_label'] >= 1.0 ? 'Anxious' : 'Relax'
        ] : null),
        "attention" => $att_data ? [
            "focus_percentage" => floatval($session['attention_percentage'] ?? $att_data['focus_percentage']),
            "attention_status" => $att_data['attention_status']
        ] : ($session['attention_percentage'] ? [
            "focus_percentage" => floatval($session['attention_percentage']),
            "attention_status" => $session['attention_cat_label'] >= 1.0 ? 'Focus' : 'Not Focus'
        ] : null),
        "timestamp" => $session['timestamp'],
        "session_id" => intval($session['session_id'])
    ]);
    
} catch (PDOException $e) {
    echo json_encode([
        "success" => false,
        "message" => "Database error: " . $e->getMessage()
    ]);
}
?>

