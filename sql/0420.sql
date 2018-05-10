-- talking4.mst_dayiphourapp
SELECT
  day, hour, ip, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct hour) uq_hour,
  count(distinct channel) uq_channel,
  count(distinct app) uq_app,
  count(distinct device) uq_device
FROM `talking2.train_test`
GROUP BY day, hour, ip

-- talking4.mst_dayhouripchannel
SELECT
  day, hour, ip, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel

-- talking4.mst_dayhouripapp
SELECT
  day, hour, ip, app, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct os) uq_os
FROM `talking2.train_test`
GROUP BY day, hour, ip, app

-- talking4.mst_dayhouripos
SELECT
  day, hour, ip, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os

-- talking4.mst_dayhouripdevice
SELECT
  day, hour, ip, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, device

###
-- talking4.mst_dayhouripappchannel
SELECT
  day, hour, ip, app, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, channel

-- talking4.mst_dayhouripchannelos
SELECT
  day, hour, ip, channel, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel, os

-- talking4.mst_dayhouripchanneldevice
SELECT
  day, hour, ip, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel, device


-- talking4.mst_dayhouripappos
SELECT
  day, hour, ip, app, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, os

-- talking4.mst_dayhouripappdevice
SELECT
  day, hour, ip, app, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, device

-- talking4.mst_dayhouriosdevice
SELECT
  day, hour, ip, device, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct app) as uq_app
FROM `talking2.train_test`
GROUP BY day, hour, ip, device, os
###

-- talking4.mst_nochannel
SELECT
  day, hour, ip, os, app, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, device

-- talking4.mst_noapp
SELECT
  day, hour, ip, os, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, channel, device

-- talking4.mst_noos
SELECT
  day, hour, ip, app, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, channel, device

-- talking4.mst_nodevice
SELECT
  day, hour, ip, os, app, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, channel
###

-- talking4.mst_all
SELECT
  day, hour, ip, os, app, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, channel, device



####

-- talking4.mst_dayhourchannel
SELECT
  day, hour, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, channel

-- talking4.mst_dayhourapp
SELECT
  day, hour, app, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, app

-- talking4.mst_dayhouros
SELECT
  day, hour, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, os

-- talking4.mst_dayhourdevice
SELECT
  day, hour, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, device


####
-- talking4.mst_app
SELECT
  app,
  count(1) as cnt,
  count(distinct channel) uq_channel
FROM `talking2.train_test`
GROUP BY app


-- talking4.mst_ip
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
FROM `talking2.train_test`
GROUP BY ip


-- talking4.mst_ipday
SELECT
  ip, day,
  count(distinct hour) uq_hour
FROM `talking2.train_test`
GROUP BY ip, day

-- talking4.mst_ipdeviceos
SELECT
  ip, device, os,
  count(distinct app) uq_app
FROM `talking2.train_test`
GROUP BY ip, device, os

####
--talking4.train_test
SELECT
min(click_id) click_id,
max(is_attributed) is_attributed,
t.day,
t.span,
t.hour,
t.minute,
t.second,
t.ip,
t.os,
t.app,
t.channel,
t.device,
t.click_time
FROM
  `talking2.train_test` as t
  GROUP BY
  day,
  span,
  hour,
  minute,
  second,
  ip,
  os,
  app,
  channel,
  device,
  click_time

-- takling4.train_test2
SELECT
  t.click_id,
  t.is_attributed ,
  t.day,
  t.span,
  t.hour,
  t.minute,
  t.second,
  t.ip,
  t.os,
  t.app,
  t.channel,
  t.device,
--  count(1) OVER(partition by t.ip, t.app, t.device, t.os, t.channel, t.click_time) as cnttt_all
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

  --STDDEV(t.hour) OVER(partition by t.ip) as stdd_1,
  --STDDEV(t.hour) OVER(partition by t.ip, t.app ) as stdd_3,
  --STDDEV(t.hour) OVER(partition by t.ip, t.channel ) as stdd_4,
  --STDDEV(t.hour) OVER(partition by t.ip, t.os, t.app, t.channel ) as stdd_15,

  a.cnt as cnt_dayiphourapp,
  --a.uq_channel as uq_channel_dayiphourapp,

  --d.cnt as cnt_dayhouripdevice,
  dd.cnt as cnt_dayhourdevice,
  ma.cnt as cnt_ma,

  o.diff as diff_dayhouripos,
  --po.diff as diff_dayhouripappos,
  --cp.diff as diff_dayhouripappchannel,
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
    `talking4.train_test` as t
  LEFT OUTER JOIN `talking4.mst_dayiphourapp` as a
  ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
  LEFT OUTER JOIN `talking4.mst_dayhouripos` as o
  ON o.day = t.day and o.hour = t.hour and o.ip = t.ip and o.os = t.os
  --
  LEFT OUTER JOIN `talking4.mst_dayhourdevice` as dd
  ON dd.day = t.day and dd.hour = t.hour and dd.device = t.device
  --
  LEFT OUTER JOIN `talking4.mst_app` as ma
  ON ma.app = t.app
  --
  LEFT OUTER JOIN `talking4.mst_ip` as mi
  ON mi.ip = t.ip
  --
  LEFT OUTER JOIN `talking4.uq_app_ipdevice` as uai
  ON uai.ip = t.ip and uai.device = t.device
  LEFT OUTER JOIN `talking4.uq_channel_iposdevice` as ucd
  ON ucd.ip = t.ip and ucd.os = t.os and ucd.device = t.device


  SELECT
  t.*,
  count(1) OVER(partition by t.ip, t.day) as cntt_ip,
  a.stdd_1, stdd_3, stdd_4,
  STDDEV(t.hour) OVER(partition by t.ip, t.os, t.app, t.channel ) as stdd_15
  FROM
  `talking4.train_test2` as t
  LEFT OUTER JOIN `talking4.stdd_1` as a
  ON a.ip = t.ip
  LEFT OUTER JOIN `talking4.stdd_3` as c
  ON c.ip = t.ip and c.app = t.app
  LEFT OUTER JOIN `talking4.stdd_4` as d
  ON d.ip = t.ip and d.channel = t.channel

SELECT
t.*,
STDDEV(t.day) OVER(partition by t.ip, t.channel) as stdday_6,

STDDEV(t.day) OVER(partition by t.ip, t.app, t.os) as stdday_7,
STDDEV(t.day) OVER(partition by t.ip, t.app, t.device) as stdday_8,
STDDEV(t.day) OVER(partition by t.ip, t.os, t.device) as stdday_10,
STDDEV(t.day) OVER(partition by t.ip, t.os, t.channel) as stdday_11,
STDDEV(t.day) OVER(partition by t.ip, t.device, t.channel) as stdday_12,

STDDEV(t.day) OVER(partition by t.ip, t.app, t.os, t.device) as stdday_13,
STDDEV(t.day) OVER(partition by t.ip, t.app, t.os, t.channel) as stdday_14,
STDDEV(t.day) OVER(partition by t.ip, t.app, t.device, t.channel) as stdday_15,
STDDEV(t.day) OVER(partition by t.ip, t.os, t.device, t.channel) as stdday_16,
STDDEV(t.day) OVER(partition by t.ip, t.app, t.os, t.device, t.channel) as stdday_17
FROM
`talking4.train_test3` as t

SELECT
t.*,
stdday_1, stdday_2, stdday_3, stdday_4, stdday_5
FROM
`talking4.train_test4` as t
LEFT OUTER JOIN `talking4.stdd_1` as a
ON a.ip = t.ip
LEFT OUTER JOIN `talking4.stdd_2` as b
ON b.ip = t.ip and b.os = t.os
LEFT OUTER JOIN `talking4.stdd_3` as c
ON c.ip = t.ip and c.app = t.app
LEFT OUTER JOIN `talking4.stdd_4` as d
ON d.ip = t.ip and d.channel = t.channel
LEFT OUTER JOIN `talking4.stdd_4` as e
ON e.ip = t.ip and e.device = t.device


  -- talking4.uq_device_ipapp
  SELECT
  t.ip, t.app, count(distinct t.device)  as uq_device_ipapp
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app

  -- talking4.uq_os_ipapp
  SELECT
  t.ip, t.app, count(distinct t.os)  as uq_os_ipapp
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app

  -- talking4.uq_channel_ipapp
  SELECT
  t.ip, t.app, count(distinct t.channel)  as uq_channel_ipapp
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app


  -- talking4.uq_app_ipdevice
  SELECT
  t.ip, t.device, count(distinct t.app)  as uq_app_ipdevice, count(1)  as cnt_app_ipdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.device

  -- talking4.uq_os_ipdevice
  SELECT
  t.ip, t.device, count(distinct t.os)  as uq_os_ipdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.device

  -- talking4.uq_channel_ipdevice
  SELECT
  t.ip, t.device, count(distinct t.channel)  as uq_channel_ipdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.device


  -- talking4.uq_app_ipos
  SELECT
  t.ip, t.os, count(distinct t.app)  as uq_app_ipos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.os

  -- talking4.uq_device_ipos
  SELECT
  t.ip, t.os, count(distinct t.device)  as uq_device_ipos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.os

  -- talking4.uq_channel_ipos
  SELECT
  t.ip, t.os, count(distinct t.channel)  as uq_channel_ipos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.os


  -- talking4.uq_app_ipchannel
  SELECT
  t.ip, t.channel, count(distinct t.app)  as uq_app_ipchannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel

  -- talking4.uq_device_ipchannel
  SELECT
  t.ip, t.channel, count(distinct t.device)  as uq_device_ipchannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel

  -- talking4.uq_os_ipchannel
  SELECT
  t.ip, t.channel, count(distinct t.os)  as uq_os_ipchannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel


  -- talking4.uq_os_ipappdevice
  SELECT
  t.ip, t.app, t.device, count(distinct t.os)  as uq_os_ipappdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.device

  -- talking4.uq_channel_ipappdevice
  SELECT
  t.ip, t.app, t.device, count(distinct t.channel)  as uq_channel_ipappdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.device


  -- talking4.uq_device_ipappos
  SELECT
  t.ip, t.app, t.os, count(distinct t.device)  as uq_device_ipappos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.os

  -- talking4.uq_channel_ipappos
  SELECT
  t.ip, t.app, t.os, count(distinct t.channel)  as uq_channel_ipappos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.os


  -- talking4.uq_device_ipappchannel
  SELECT
  t.ip, t.app, t.channel, count(distinct t.device)  as uq_device_ipappchannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.channel

  -- talking4.uq_os_ipappchannel
  SELECT
  t.ip, t.app, t.channel, count(distinct t.os)  as uq_os_ipappchannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.channel


  -- talking4.uq_app_iposdevice
  SELECT
  t.ip, t.os, t.device, count(distinct t.app)  as uq_app_iposdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.os, t.device

  -- talking4.uq_channel_iposdevice
  SELECT
  t.ip, t.os, t.device, count(distinct t.channel)  as uq_channel_iposdevice, count(1) as cnt_channel_iposdevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.os, t.device


  -- talking4.uq_app_ipchanneldevice
  SELECT
  t.ip, t.channel, t.device, count(distinct t.app)  as uq_app_ipchanneldevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel, t.device

  -- talking4.uq_os_ipchanneldevice
  SELECT
  t.ip, t.channel, t.device, count(distinct t.os)  as uq_os_ipchanneldevice
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel, t.device


  -- talking4.uq_app_ipchannelos
  SELECT
  t.ip, t.channel, t.os, count(distinct t.app)  as uq_app_ipchannelos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel, t.os

  -- talking4.uq_device_ipchannelos
  SELECT
  t.ip, t.channel, t.os, count(distinct t.device)  as uq_device_ipchannelos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel, t.os

  -- talking4.uq_channel_ipaoodeviceos
  SELECT
  t.ip, t.app, t.device, t.os, count(distinct t.channel)  as uq_channel_ipaoodeviceos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.device, t.os

  -- talking4.uq_app_ipchanneldeviceos
  SELECT
  t.ip, t.channel, t.device, t.os, count(distinct t.app)  as uq_app_ipchanneldeviceos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.channel, t.device, t.os

  -- talking4.uq_device_ipappchannelos
  SELECT
  t.ip, t.app, t.channel, t.os, count(distinct t.device)  as uq_device_ipappchannelos
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.channel, t.os

  -- talking4.uq_os_ipappdevicechannel
  SELECT
  t.ip, t.app, t.device, t.channel, count(distinct t.os)  as uq_os_ipappdevicechannel
  FROM `talking2.train_test` as t
  GROUP BY t.ip, t.app, t.device, t.channel

  -- talking4.stdd_1
  SELECT
  ip, STDDEV(hour) stdd_1, AVG(hour) avgd_1, STDDEV(day) stdday_1
  FROM
    `talking2.train_test`
  GROUP BY ip

  -- talking4.stdd_2
  SELECT
  ip, os, STDDEV(hour) stdd_2, STDDEV(day) stdday_2
  FROM
    `talking2.train_test`
  GROUP BY ip, os

  -- talking4.stdd_3
  SELECT
  ip, app, STDDEV(hour) stdd_3, AVG(hour) avgd_3, STDDEV(day) stdday_3
  FROM
    `talking2.train_test`
  GROUP BY ip, app

  -- talking4.stdd_4
  SELECT
  ip, channel, STDDEV(hour) stdd_4, AVG(hour) avgd_4, STDDEV(day) stdday_4
  FROM
    `talking2.train_test`
  GROUP BY ip, channel

  -- talking4.stdd_5
  SELECT
  ip, device, STDDEV(hour) stdd_5, STDDEV(day) stdday_5
  FROM
    `talking2.train_test`
  GROUP BY ip, device
