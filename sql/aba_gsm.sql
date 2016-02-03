/*
Navicat MySQL Data Transfer

Source Server         : local
Source Server Version : 50624
Source Host           : localhost:3306
Source Database       : network

Target Server Type    : MYSQL
Target Server Version : 50624
File Encoding         : 65001

Date: 2016-01-28 16:03:47
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for aba_gsm
-- ----------------------------
DROP TABLE IF EXISTS `aba_gsm`;
CREATE TABLE `aba_gsm` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `from_id` int(11) DEFAULT NULL,
  `to_id` int(11) DEFAULT NULL,
  `date_index` int(11) DEFAULT NULL,
  `date_time` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
  KEY `date_index` (`date_index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
