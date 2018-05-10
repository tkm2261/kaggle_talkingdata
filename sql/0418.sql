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
  count(distinct channel) uq_channel,
  count(distinct app) uq_app,
  count(distinct device) uq_device,
  count(distinct os) uq_os
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

-- takling4.train_test2
SELECT
  t.click_id,
  t.is_attributed ,
  t.day,
  t.span,
  t.hour,
  t.ip,
  t.os,
  t.app,
  t.channel,
  t.device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time) as row_ip,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time) as row_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device order by click_time) as row_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app order by click_time) as row_os_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time) as row_no_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time desc) as row_ip_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time desc) as row_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device order by click_time desc) as row_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app order by click_time desc) as row_os_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time desc) as row_no_channel_r,
uq_app_ipdevice,
uq_app_iposdevice,
uq_os_ipchannel,
uq_os_ipapp,
uq_app_ipchannel,
uq_channel_ipdevice,
uq_channel_iposdevice,
uq_os_ipdevice,
uq_os_ipappdevice,
uq_device_ipchannel,
uq_channel_ipos,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.hour, t.app, t.device order by click_time desc), SECOND) as nextclick_1,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.hour order by click_time desc), SECOND) as nextclick_3,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app order by click_time desc), SECOND) as nextclick_2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os order by click_time desc), SECOND) as nextclick_4,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.device order by click_time desc), SECOND) as nextclick_5,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.channel order by click_time desc), SECOND) as nextclick_6,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.os order by click_time desc), SECOND) as nextclick_7,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.device order by click_time desc), SECOND) as nextclick_8,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.channel order by click_time desc), SECOND) as nextclick_9,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os, t.device order by click_time desc), SECOND) as nextclick_10,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os, t.channel order by click_time desc), SECOND) as nextclick_11,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.device, t.channel order by click_time desc), SECOND) as nextclick_12,

  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.os, t.device order by click_time desc), SECOND) as nextclick_13,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.os, t.channel order by click_time desc), SECOND) as nextclick_14,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.device, t.channel order by click_time desc), SECOND) as nextclick_15,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.os, t.device, t.channel order by click_time desc), SECOND) as nextclick_16,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.ip, t.day, t.app, t.os, t.device, t.channel order by click_time desc), SECOND) as nextclick_17,


  STDDEV(t.hour) OVER(partition by t.ip, t.day, t.channel) std_ipdaychannel,
  STDDEV(t.hour) OVER(partition by t.ip, t.app, t.os) std_ipappos,
  STDDEV(t.hour) OVER(partition by t.ip, t.app, t.channel) std_ipappchannel,
  STDDEV(t.day) OVER(partition by t.ip, t.app, t.channel) std_day_ipappchannel,
  AVG(t.day) OVER(partition by t.ip, t.app, t.channel) avg_ipdaychannel,

  a.cnt as cnt_dayiphourapp,
  a.uq_hour as uq_hour_dayiphourapp,
  a.uq_channel as uq_channel_dayiphourapp,
  a.uq_app as uq_app_dayiphourapp,
  a.uq_device as uq_device_dayiphourapp,

  p.cnt as cnt_dayhouripapp,
  o.cnt as cnt_dayhouripos,
  d.cnt as cnt_dayhouripdevice,
  dd.cnt as cnt_dayhourdevice,
  po.cnt as cnt_dayhouripappos,
  od.cnt as cnt_dayhouriosdevice,
  nc.cnt as cnt_nochannel,
  ma.cnt as cnt_ma,
  ma.uq_channel as uq_channel_ma,

  o.diff as diff_dayhouripos,
  po.diff as diff_dayhouripappos,
  cp.diff as diff_dayhouripappchannel,

  mi.uq_channel as uq_channel_ip,
  mi.uq_app as uq_app_ip,
  mi.uq_device as uq_device_ip,
  mi.uq_os as uq_os_ip,
  mid.uq_hour as uq_hour_ipday,
  mido.uq_app as uq_app_ipdeviceos
FROM
  `talking2.train_test` as t
LEFT OUTER JOIN `talking4.mst_dayiphourapp` as a
ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayhouripapp` as p
ON p.day = t.day and p.hour = t.hour and p.ip = t.ip and p.app = t.app
LEFT OUTER JOIN `talking4.mst_dayhouripos` as o
ON o.day = t.day and o.hour = t.hour and o.ip = t.ip and o.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhouripdevice` as d
ON d.day = t.day and d.hour = t.hour and d.ip = t.ip and d.device = t.device
--
LEFT OUTER JOIN `talking4.mst_dayhouripappchannel` as cp
ON cp.day = t.day and cp.hour = t.hour and cp.ip = t.ip and cp.app = t.app and cp.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripappos` as po
ON po.day = t.day and po.hour = t.hour and po.ip = t.ip and po.app = t.app and po.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhouriosdevice` as od
ON od.day = t.day and od.hour = t.hour and od.ip = t.ip and od.os = t.os and od.device = t.device
--
LEFT OUTER JOIN `talking4.mst_nochannel` as nc
ON nc.day = t.day and nc.hour = t.hour and nc.ip = t.ip
and nc.os = t.os and nc.app = t.app and nc.device = t.device
--
LEFT OUTER JOIN `talking4.mst_dayhourdevice` as dd
ON dd.day = t.day and dd.hour = t.hour and dd.device = t.device
--
LEFT OUTER JOIN `talking4.mst_app` as ma
ON ma.app = t.app
--
LEFT OUTER JOIN `talking4.mst_ip` as mi
ON mi.ip = t.ip
LEFT OUTER JOIN `talking4.mst_ipday` as mid
ON mid.ip = t.ip and mid.day = t.day
LEFT OUTER JOIN `talking4.mst_ipdeviceos` as mido
ON mido.ip = t.ip and mido.device = t.device and mido.os = t.os
--
LEFT OUTER JOIN `talking4.uq_device_ipapp` as udi
ON udi.ip = t.ip and udi.app = t.app
LEFT OUTER JOIN `talking4.uq_os_ipapp` as uoi
ON uoi.ip = t.ip and uoi.app = t.app
LEFT OUTER JOIN `talking4.uq_channel_ipapp` as uci
ON uci.ip = t.ip and uci.app = t.app
LEFT OUTER JOIN `talking4.uq_app_ipdevice` as uai
ON uai.ip = t.ip and uai.device = t.device
LEFT OUTER JOIN `talking4.uq_os_ipdevice` as uop
ON uop.ip = t.ip and uop.device = t.device
LEFT OUTER JOIN `talking4.uq_channel_ipdevice` as ucp
ON ucp.ip = t.ip and ucp.device = t.device
LEFT OUTER JOIN `talking4.uq_app_ipos` as uap
ON uap.ip = t.ip and uap.os = t.os
LEFT OUTER JOIN `talking4.uq_device_ipos` as udp
ON udp.ip = t.ip and udp.os = t.os
LEFT OUTER JOIN `talking4.uq_channel_ipos` as uct
ON uct.ip = t.ip and uct.os = t.os
LEFT OUTER JOIN `talking4.uq_app_ipchannel` as uac
ON uac.ip = t.ip and uac.channel = t.channel
LEFT OUTER JOIN `talking4.uq_device_ipchannel` as udc
ON udc.ip = t.ip and udc.channel = t.channel
LEFT OUTER JOIN `talking4.uq_os_ipchannel` as uoc
ON uoc.ip = t.ip and uoc.channel = t.channel
LEFT OUTER JOIN `talking4.uq_os_ipappdevice` as uod
ON uod.ip = t.ip and uod.app = t.app and uod.device = t.device
LEFT OUTER JOIN `talking4.uq_channel_ipappdevice` as uch
ON uch.ip = t.ip and uch.app = t.app and uch.device = t.device
LEFT OUTER JOIN `talking4.uq_device_ipappos` as uds
ON uds.ip = t.ip and uds.app = t.app and uds.os = t.os
LEFT OUTER JOIN `talking4.uq_channel_ipappos` as ucs
ON ucs.ip = t.ip and ucs.app = t.app and ucs.os = t.os
LEFT OUTER JOIN `talking4.uq_device_ipappchannel` as udl
ON udl.ip = t.ip and udl.app = t.app and udl.channel = t.channel
LEFT OUTER JOIN `talking4.uq_os_ipappchannel` as uol
ON uol.ip = t.ip and uol.app = t.app and uol.channel = t.channel
LEFT OUTER JOIN `talking4.uq_app_iposdevice` as uam
ON uam.ip = t.ip and uam.os = t.os and uam.device = t.device
LEFT OUTER JOIN `talking4.uq_channel_iposdevice` as ucd
ON ucd.ip = t.ip and ucd.os = t.os and ucd.device = t.device
LEFT OUTER JOIN `talking4.uq_app_ipchanneldevice` as uad
ON uad.ip = t.ip and uad.channel = t.channel and uad.device = t.device
LEFT OUTER JOIN `talking4.uq_os_ipchanneldevice` as uov
ON uov.ip = t.ip and uov.channel = t.channel and uov.device = t.device
LEFT OUTER JOIN `talking4.uq_app_ipchannelos` as ual
ON ual.ip = t.ip and ual.channel = t.channel and ual.os = t.os
LEFT OUTER JOIN `talking4.uq_device_ipchannelos` as udv
ON udv.ip = t.ip and udv.channel = t.channel and udv.os = t.os
LEFT OUTER JOIN `talking4.uq_channel_ipaoodeviceos` as uce
ON uce.ip = t.ip and uce.app = t.app and uce.device = t.device and uce.os = t.os
LEFT OUTER JOIN `talking4.uq_app_ipchanneldeviceos` as uae
ON uae.ip = t.ip and uae.channel = t.channel and uae.device = t.device and uae.os = t.os
LEFT OUTER JOIN `talking4.uq_device_ipappchannelos` as ude
ON ude.ip = t.ip and ude.channel = t.channel and ude.app = t.app and ude.os = t.os
LEFT OUTER JOIN `talking4.uq_os_ipappdevicechannel` as uoe
ON uoe.ip = t.ip and uoe.app = t.app and uoe.device = t.device and uoe.channel = t.channel



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
  t.ip, t.device, count(distinct t.app)  as uq_app_ipdevice
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
  t.ip, t.os, t.device, count(distinct t.channel)  as uq_channel_iposdevice
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
