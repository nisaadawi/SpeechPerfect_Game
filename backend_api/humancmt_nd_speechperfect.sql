-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Dec 10, 2025 at 04:08 AM
-- Server version: 10.3.39-MariaDB-log-cll-lve
-- PHP Version: 8.1.33

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `humancmt_nd_speechperfect`
--

-- --------------------------------------------------------

--
-- Table structure for table `pre_questionnaire`
--

CREATE TABLE `pre_questionnaire` (
  `pre_id` int(11) NOT NULL,
  `id` int(11) DEFAULT NULL,
  `submission_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `ans_q1` int(11) DEFAULT NULL,
  `ans_q2` int(11) DEFAULT NULL,
  `ans_q3` int(11) DEFAULT NULL,
  `ans_q4` int(11) DEFAULT NULL,
  `ans_q5` int(11) DEFAULT NULL,
  `ans_q6` int(11) DEFAULT NULL,
  `ans_q7` int(11) DEFAULT NULL,
  `ans_q8` int(11) DEFAULT NULL,
  `ans_q9` text DEFAULT NULL,
  `ans_q10` int(11) DEFAULT NULL,
  `ans_q11` int(11) DEFAULT NULL,
  `ans_q12` int(11) DEFAULT NULL,
  `ans_q13` int(11) DEFAULT NULL,
  `ans_q14` int(11) DEFAULT NULL,
  `ans_q15` int(11) DEFAULT NULL,
  `ans_q16` int(11) DEFAULT NULL,
  `ans_q17` int(11) DEFAULT NULL,
  `ans_q18` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_attention`
--

CREATE TABLE `tbl_attention` (
  `attention_id` int(11) NOT NULL,
  `id` int(11) DEFAULT NULL,
  `focus_percentage` int(20) NOT NULL,
  `attention_status` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_audio`
--

CREATE TABLE `tbl_audio` (
  `audio_id` int(11) NOT NULL,
  `id` int(11) DEFAULT NULL,
  `audio_file` varchar(300) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_heart_rate`
--

CREATE TABLE `tbl_heart_rate` (
  `hr_id` int(11) NOT NULL,
  `id` int(11) DEFAULT NULL,
  `hr_min` int(11) NOT NULL,
  `hr_max` int(11) NOT NULL,
  `hr_avg` int(11) NOT NULL,
  `hr_status` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(100) NOT NULL,
  `age` int(11) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `age`, `gender`, `email`, `password`, `created_at`) VALUES
(1, 'nisaaei', 22, 'girl', 'wereen0909@gmail.com', '$2y$10$rTh4kkCaCGpq5BzDbka3IeUD0ytNBmVxLMWWkuLHe3YrYJ3UI0TTS', '2025-07-05 09:32:56'),
(2, 'wereenz', 22, 'female', 'nisaadawi1004', '$2y$10$aKm9CNqmLZQsO9Av/PxqjO5LIc4ZWYvKPb1ggBzws6OFlahz8ze0q', '2025-07-05 09:36:32'),
(3, 'wereenz', 22, 'female', 'nisaadawi1004@gmail.com', '$2y$10$TGWltUI5OUOvdXWrnGxESeMUmYJSugym9ZxOTSpF3y15..EFTi6fm', '2025-07-05 09:54:35'),
(5, 'chunsa1004', 21, 'female', 'chunsa1004@gmail.com', '$2y$10$EncShqXs38JoETUAmD8JI.YkdEP6RKOLb8tjnkKw7hSp7vzBH8ImS', '2025-07-05 15:05:08'),
(6, 'chunser', 21, 'female', 'nd@gmail.com', '$2y$10$xFx/mbGcJaJcRnHrsLvs0u4eXTndGv0KplD7yFatADXm7dX/en7xi', '2025-07-05 15:08:10'),
(7, 'seicha', 21, 'female', 'weee@gmail.com', '$2y$10$WGedDDRzMpj3WWgABOC1y.CDKllrRX/T97eHeG.E563NhTkL6e9bu', '2025-07-05 15:39:07'),
(8, 'nisaawi', 21, 'female', 'we@gmail.com', '$2y$10$NcF9t9U9.NcLgCL/08/AEuGq5NdcK.KBmCVnR8sjmpKZCE6Thhq2C', '2025-07-05 15:41:31'),
(9, 'wereen', 21, 'female', 'cheche@gmail.com', '$2y$10$CU17whmmWJe6n5BGycz8F.hYfTxbUb1PD/uSE0PbrnW1toJcQpUqK', '2025-07-05 17:17:58'),
(10, 'nisaawi', 21, 'female', 'chuchu@gmail.com', '$2y$10$SdOW52GnqjCmznWWGnHWa.wvuE6vpgoowvuK0klaw8S6cEfXoaVvi', '2025-07-05 17:22:09'),
(11, 'wereen', 21, 'female', 'nisaa123@gmail.com', '$2y$10$c3GBZ92wVGGAQ0PAdpTEDOJn.D16xQNEK6vJWtuiZXgQ8rpLWoMEG', '2025-07-06 04:01:15'),
(12, 'xxx', 12, 'xx', 'xxx', '$2y$10$rYoAK/rAdgfQNtQmc9ITgO7sIY9/ck1MuAvo.J3Xz3DWVTSnbVdOy', '2025-10-08 14:44:53'),
(13, 'nwho_n', 22, 'Female', 'abumonza@gmail.com', 'abumonza', '2025-11-06 06:26:19');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `pre_questionnaire`
--
ALTER TABLE `pre_questionnaire`
  ADD PRIMARY KEY (`pre_id`),
  ADD KEY `id` (`id`);

--
-- Indexes for table `tbl_attention`
--
ALTER TABLE `tbl_attention`
  ADD KEY `fk_attention_id` (`id`);

--
-- Indexes for table `tbl_audio`
--
ALTER TABLE `tbl_audio`
  ADD PRIMARY KEY (`audio_id`),
  ADD KEY `fk_audio_id` (`id`);

--
-- Indexes for table `tbl_heart_rate`
--
ALTER TABLE `tbl_heart_rate`
  ADD PRIMARY KEY (`hr_id`),
  ADD KEY `fk_hr_id` (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `pre_questionnaire`
--
ALTER TABLE `pre_questionnaire`
  MODIFY `pre_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `tbl_audio`
--
ALTER TABLE `tbl_audio`
  MODIFY `audio_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `tbl_heart_rate`
--
ALTER TABLE `tbl_heart_rate`
  MODIFY `hr_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `pre_questionnaire`
--
ALTER TABLE `pre_questionnaire`
  ADD CONSTRAINT `pre_questionnaire_ibfk_1` FOREIGN KEY (`id`) REFERENCES `users` (`id`);

--
-- Constraints for table `tbl_attention`
--
ALTER TABLE `tbl_attention`
  ADD CONSTRAINT `fk_attention_id` FOREIGN KEY (`id`) REFERENCES `users` (`id`);

--
-- Constraints for table `tbl_audio`
--
ALTER TABLE `tbl_audio`
  ADD CONSTRAINT `fk_audio_id` FOREIGN KEY (`id`) REFERENCES `users` (`id`);

--
-- Constraints for table `tbl_heart_rate`
--
ALTER TABLE `tbl_heart_rate`
  ADD CONSTRAINT `fk_hr_id` FOREIGN KEY (`id`) REFERENCES `users` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
