-- takling.train2
SELECT
*,
TIMESTAMP_DIFF(click_time, LAG(click_time) OVER(partition by ip order by click_time), SECOND) as timediff,
EXTRACT(year from click_time) as year,
EXTRACT(month from click_time) as month,
EXTRACT(day from click_time) as day,
EXTRACT(DAYOFWEEK from click_time) as dayofweek,
EXTRACT(HOUR from click_time) as hour
FROM
  `talking.train`

-- takling.test2
SELECT
*,
TIMESTAMP_DIFF(click_time, LAG(click_time) OVER(partition by ip order by click_time), SECOND) as timediff,
EXTRACT(year from click_time) as year,
EXTRACT(month from click_time) as month,
EXTRACT(day from click_time) as day,
EXTRACT(DAYOFWEEK from click_time) as dayofweek,
EXTRACT(HOUR from click_time) as hour
FROM
  `talking.test`

-- takling.train_test
SELECT
null as click_id,
ip, app, device, os, channel, click_time, attributed_time, is_attributed, timediff, year, month, day, dayofweek, hour
FROM
`talking.train2`
UNION ALL
SELECT
click_id,
ip, app, device, os, channel, click_time, null as attributed_time, null as is_attributed, timediff, year, month, day, dayofweek, hour
FROM
`talking.test2`

-- takling.train_test2
SELECT
  *,
  sum(cast(is_attributed as int64)) OVER(partition by ip order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as sum_attr,
  TIMESTAMP_DIFF(click_time, MAX(attributed_time) OVER(partition by ip order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), SECOND) as last_attr,
FROM
  `talking.train_test`

-- dmt_train
SELECT
  ip,
  concat('[', STRING_AGG(cast(is_attributed as string) order by click_time), ']') as list_target,
  concat('[', STRING_AGG(cast(app as string) order by click_time), ']') as list_app,
  concat('[', STRING_AGG(cast(device as string) order by click_time), ']') as list_device,
  concat('[', STRING_AGG(cast(os as string) order by click_time), ']') as list_os,
  concat('[', STRING_AGG(cast(channel as string) order by click_time), ']') as list_ch,
  concat('[', STRING_AGG(cast(timediff as string) order by click_time), ']') as list_timediff,
  concat('[', STRING_AGG(cast(month as string) order by click_time), ']') as list_month,
  concat('[', STRING_AGG(cast(day as string) order by click_time), ']') as list_day,
  concat('[', STRING_AGG(cast(dayofweek as string) order by click_time), ']') as list_dayofweek,
  concat('[', STRING_AGG(cast(hour as string) order by click_time), ']') as list_hour,
  concat('[', STRING_AGG(cast(sum_attr as string) order by click_time), ']') as list_sum_attr,
  concat('[', STRING_AGG(cast(last_attr as string) order by click_time), ']') as list_attr
FROM
  `talking.train_test2`
WHERE
  click_id is null
group by
  ip

-- dmt_test
SELECT
  ip,
  concat('[', STRING_AGG(cast(click_id as string) order by click_time), ']') as list_click_id,
  concat('[', STRING_AGG(cast(app as string) order by click_time), ']') as list_app,
  concat('[', STRING_AGG(cast(device as string) order by click_time), ']') as list_device,
  concat('[', STRING_AGG(cast(os as string) order by click_time), ']') as list_os,
  concat('[', STRING_AGG(cast(channel as string) order by click_time), ']') as list_ch,
  concat('[', STRING_AGG(cast(timediff as string) order by click_time), ']') as list_timediff,
  concat('[', STRING_AGG(cast(month as string) order by click_time), ']') as list_month,
  concat('[', STRING_AGG(cast(day as string) order by click_time), ']') as list_day,
  concat('[', STRING_AGG(cast(dayofweek as string) order by click_time), ']') as list_dayofweek,
  concat('[', STRING_AGG(cast(hour as string) order by click_time), ']') as list_hour,
  concat('[', STRING_AGG(cast(sum_attr as string) order by click_time), ']') as list_sum_attr,
  concat('[', STRING_AGG(cast(last_attr as string) order by click_time), ']') as list_attr
FROM
  `talking.train_test2`
WHERE
  click_id is not null
group by
  ip
