-- Feature engineering BigQuery SQL queries for the kaggle talkingdata competition by tkm2261
-- it may acheve 0.9823 on the public LB with simple GBDT.

-- destination table: takling.train_test
SELECT
null as click_id,
ip, app, device, os, channel, click_time, attributed_time, is_attributed, timediff, year, month, day, dayofweek, hour, minute, second
FROM
`talking.train`
UNION ALL
SELECT
click_id,
ip, app, device, os, channel, click_time, null as attributed_time, null as is_attributed, timediff, year, month, day, dayofweek, hour, minute, second
FROM
`talking.test`

-- destination table: takling.mst_dayhouripos
SELECT
  day, hour, ip, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking.train_test`
GROUP BY day, hour, ip, os

-- destination table: takling.mst_dayiphourapp
SELECT
  day, hour, ip, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct hour) uq_hour,
  count(distinct channel) uq_channel,
  count(distinct app) uq_app,
  count(distinct device) uq_device
FROM `talking.train_test`
GROUP BY day, hour, ip

-- destination table: takling.mst_dayhourdevice
SELECT
  day, hour, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking.train_test`
GROUP BY day, hour, device


-- destination table: takling.mst_app
SELECT
  app,
  count(1) as cnt,
  count(distinct channel) uq_channel
FROM `talking.train_test`
GROUP BY app

-- destination table: takling.mst_ip
SELECT
  ip,
  count(1) cnt_ip,
  count(distinct channel) uq_channel,
  count(distinct app) uq_app,
  count(distinct device) uq_device,
  count(distinct os) uq_os,
  STDDEV(UNIX_MICROS(click_time) / 1000) std_ip_time,
  STDDEV(hour) std_ip_hour,
  STDDEV(minute) std_ip_minute,
  STDDEV(second) std_ip_second
FROM `talking.train_test`
GROUP BY ip

-- destination table: takling.uq_app_ipdevice
SELECT
t.ip, t.device, count(distinct t.app)  as uq_app_ipdevice, count(1)  as cnt_app_ipdevice
FROM `talking.train_test` as t
GROUP BY t.ip, t.device

-- destination table: talking.uq_channel_iposdevice
SELECT
t.ip, t.os, t.device, count(distinct t.channel)  as uq_channel_iposdevice, count(1) as cnt_channel_iposdevice
FROM `talking2.train_test` as t
GROUP BY t.ip, t.os, t.device

-- destination table: takling.train_test2
SELECT
  t.click_id,
  t.is_attributed ,
  t.day,
  t.hour,
  t.minute,
  t.second,
  t.ip,
  t.os,
  t.app,
  t.channel,
  t.device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time) as row_ip,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time desc) as row_ip_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time) as row_no_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time desc) as row_no_channel_r,

  uq_app_ipdevice,
  uq_channel_iposdevice,
  cnt_app_ipdevice,
  cnt_channel_iposdevice,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app order by click_time desc), SECOND) as nextclick_2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os order by click_time desc), SECOND) as nextclick_4,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.os, t.device order by click_time desc), SECOND) as nextclick_13,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by t.ip, t.day, t.app order by click_time desc), SECOND) as nextclick_2_2,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app order by click_time), SECOND) as nextclick_2_b,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os order by click_time), SECOND) as nextclick_4_b,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.channel order by click_time), SECOND) as nextclick_6_b,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip,  t.app order by click_time desc), SECOND) as nextclick_2_n,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip,  t.os order by click_time desc), SECOND) as nextclick_4_n,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip,  t.channel order by click_time desc), SECOND) as nextclick_6_n,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip,  t.app, t.os order by click_time desc), SECOND) as nextclick_7_n,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip,  t.app, t.os, t.device order by click_time desc), SECOND) as nextclick_13_n,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by t.ip,  t.app, t.os, t.device order by click_time desc), SECOND) as nextclick_13_n2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by t.ip,  t.app, t.os, t.device order by click_time desc), SECOND) as nextclick_13_n3,

  STDDEV(UNIX_MICROS(click_time) / 1000) OVER(partition by t.ip,  t.app) as stdtime_1,
  STDDEV(UNIX_MICROS(click_time) / 1000) OVER(partition by t.ip,  t.os) as stdtime_2,
  STDDEV(UNIX_MICROS(click_time) / 1000) OVER(partition by t.ip,  t.channel) as stdtime_3,

  TIMESTAMP_DIFF(MAX(click_time) OVER(partition by t.ip, t.os), MIN(click_time) OVER(partition by t.ip, t.os), SECOND) as dur_2,
  TIMESTAMP_DIFF(MAX(click_time) OVER(partition by t.ip, t.channel), MIN(click_time) OVER(partition by t.ip,  t.channel), SECOND) as dur_3,

  STDDEV(t.day) OVER(partition by t.ip, t.app, t.channel) std_day_ipappchannel,

  STDDEV(t.hour) OVER(partition by t.ip, t.day ) as std_1,
  STDDEV(t.hour) OVER(partition by t.ip, t.day, t.app ) as std_3,
  STDDEV(t.hour) OVER(partition by t.ip, t.day, t.channel ) as std_ipdaychannel,
  STDDEV(t.hour) OVER(partition by t.ip, t.day, t.os, t.device ) as std_8,

  a.cnt as cnt_dayiphourapp,

  dd.cnt as cnt_dayhourdevice,
  ma.cnt as cnt_ma,

  o.diff as diff_dayhouripos,

  mi.cnt_ip,
  mi.uq_channel as uq_channel_ip,
  mi.uq_app as uq_app_ip,
  mi.uq_device as uq_device_ip,
  mi.uq_os as uq_os_ip,
  mi.std_ip_time,
  mi.std_ip_hour,
  mi.std_ip_minute,
  mi.std_ip_second
FROM
  `talking.train_test` as t
LEFT OUTER JOIN `talking.mst_dayiphourapp` as a
ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
LEFT OUTER JOIN `talking.mst_dayhouripos` as o
ON o.day = t.day and o.hour = t.hour and o.ip = t.ip and o.os = t.os
--
LEFT OUTER JOIN `talking.mst_dayhourdevice` as dd
ON dd.day = t.day and dd.hour = t.hour and dd.device = t.device
--
LEFT OUTER JOIN `talking.mst_app` as ma
ON ma.app = t.app
--
LEFT OUTER JOIN `talking.mst_ip` as mi
ON mi.ip = t.ip
--
LEFT OUTER JOIN `talking.uq_app_ipdevice` as uai
ON uai.ip = t.ip and uai.device = t.device
LEFT OUTER JOIN `talking.uq_channel_iposdevice` as ucd
ON ucd.ip = t.ip and ucd.os = t.os and ucd.device = t.device


-- destination table: takling.stdd_1
SELECT
ip, STDDEV(hour) stdd_1, AVG(hour) avgd_1, STDDEV(day) stdday_1
FROM
  `talking.train_test`
GROUP BY ip


-- destination table: takling.stdd_3
SELECT
ip, app, STDDEV(hour) stdd_3, AVG(hour) avgd_3, STDDEV(day) stdday_3
FROM
  `talking.train_test`
GROUP BY ip, app

-- destination table: takling.stdd_4
SELECT
ip, channel, STDDEV(hour) stdd_4, AVG(hour) avgd_4, STDDEV(day) stdday_4
FROM
  `talking.train_test`
GROUP BY ip, channel

-- destination table: takling.datamart
SELECT
t.*,
count(1) OVER(partition by t.ip, t.day) as cntt_ip,
a.stdd_1, stdd_3, stdd_4,
STDDEV(t.hour) OVER(partition by t.ip, t.os, t.app, t.channel ) as stdd_15
FROM
`talking.train_test2` as t
LEFT OUTER JOIN `talking.stdd_1` as a
ON a.ip = t.ip
LEFT OUTER JOIN `talking.stdd_3` as c
ON c.ip = t.ip and c.app = t.app
LEFT OUTER JOIN `talking.stdd_4` as d
ON d.ip = t.ip and d.channel = t.channel

-- destination table: takling.train_data
SELECT
*
FROM
`talking4.train_test3`
WHERE
  day <= 8

-- destination table: takling.valid_data
SELECT
*
FROM
`talking4.train_test3`
WHERE
  day = 9

-- destination table: takling.test_data
SELECT
*
FROM
`talking4.train_test3`
WHERE
  day = 10
