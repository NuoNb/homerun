/**
 * Data Lab → Topics panel.
 *
 * The canonical "what data exists in this system" view.  Reads from
 * /api/topics — the recorded-event-bus topic catalog — and replaces
 * the previous fragmented view (data_sources + provider_datasets +
 * scattered per-recorder UIs) with one uniform table.
 *
 * Operators see, per topic:
 *   - slug + title + description
 *   - storage backing (parquet / sql_table / memory)
 *   - publishers (which services emit it) and subscribers (which
 *     strategies read it)
 *   - retention + replayability + enable/disable state
 *   - last published / last replayed timestamps + cumulative volume
 *
 * Single-click: topic detail drawer with replay-preview button →
 * shows the next 25 envelopes in the requested time window so the
 * operator can sanity-check schema before kicking off a 1-hour
 * backtest.
 */
import { useMemo, useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { formatDistanceToNow } from 'date-fns'
import {
  Loader2,
  RefreshCw,
  Play,
  Power,
  Trash2,
  HardDrive,
} from 'lucide-react'

import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Input } from './ui/input'
import { ScrollArea } from './ui/scroll-area'
import {
  listTopics,
  replayPreview,
  patchTopic,
  getBusRotationSettings,
  updateBusRotationSettings,
  triggerPruneNow,
  type TopicSpec,
  type StorageKind,
  topicStorageBadgeColor,
} from '../services/apiTopicCatalog'
import { cn } from '../lib/utils'

function formatTopicVolume(spec: TopicSpec, t: (key: string, opts?: Record<string, unknown>) => string): string {
  if (spec.event_count <= 0 && spec.bytes_on_disk <= 0) return '—'
  const count = spec.event_count >= 1_000_000
    ? `${(spec.event_count / 1_000_000).toFixed(1)}M`
    : spec.event_count >= 1_000
    ? `${(spec.event_count / 1_000).toFixed(1)}K`
    : `${spec.event_count}`
  const events = t('dataLabTopics.volumeEvents', { n: count })
  const sizeMB = spec.bytes_on_disk / (1024 * 1024)
  if (sizeMB <= 0) return events
  if (sizeMB >= 1024) return `${events} · ${(sizeMB / 1024).toFixed(2)} GB`
  if (sizeMB >= 1) return `${events} · ${sizeMB.toFixed(1)} MB`
  return `${events} · ${(sizeMB * 1024).toFixed(0)} KB`
}

type StorageFilter = 'all' | StorageKind

export default function DataLabTopics() {
  const { t } = useTranslation()
  const [storageFilter, setStorageFilter] = useState<StorageFilter>('all')
  const [enabledOnly, setEnabledOnly] = useState(false)
  const [replayableOnly, setReplayableOnly] = useState(false)
  const [selected, setSelected] = useState<TopicSpec | null>(null)

  const query = useQuery({
    queryKey: ['topic-catalog', storageFilter, enabledOnly, replayableOnly],
    queryFn: () =>
      listTopics({
        storage_kind: storageFilter === 'all' ? undefined : storageFilter,
        enabled_only: enabledOnly,
        replayable_only: replayableOnly,
      }),
    refetchInterval: 30_000,
  })

  const topics: TopicSpec[] = query.data ?? []
  const stats = useMemo(() => {
    const total = topics.length
    const totalEvents = topics.reduce((s: number, t: TopicSpec) => s + t.event_count, 0)
    const totalBytes = topics.reduce((s: number, t: TopicSpec) => s + t.bytes_on_disk, 0)
    const byKind = topics.reduce<Record<string, number>>((acc, t: TopicSpec) => {
      acc[t.storage_kind] = (acc[t.storage_kind] ?? 0) + 1
      return acc
    }, {})
    return { total, totalEvents, totalBytes, byKind }
  }, [topics])

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      {/* Header / filters */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">{t('dataLabTopics.title')}</h3>
          <p className="text-[11px] text-muted-foreground">
            {t('dataLabTopics.subtitle')}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* Storage filter pills */}
          <div className="flex rounded-md border border-border/40 bg-card/40 p-0.5">
            {(['all', 'parquet', 'sql_table', 'memory'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setStorageFilter(k)}
                className={cn(
                  'rounded-sm px-2.5 py-1 text-[11px] font-medium transition-colors',
                  storageFilter === k
                    ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {k === 'all' ? t('dataLabTopics.filterAll') : k}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1 text-[11px]">
            <input
              type="checkbox"
              checked={enabledOnly}
              onChange={(e) => setEnabledOnly(e.target.checked)}
              className="h-3 w-3"
            />
            {t('dataLabTopics.enabledOnly')}
          </label>
          <label className="flex items-center gap-1 text-[11px]">
            <input
              type="checkbox"
              checked={replayableOnly}
              onChange={(e) => setReplayableOnly(e.target.checked)}
              className="h-3 w-3"
            />
            {t('dataLabTopics.replayableOnly')}
          </label>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1 text-[10px]"
            onClick={() => query.refetch()}
            disabled={query.isFetching}
          >
            <RefreshCw className={cn('h-3 w-3', query.isFetching && 'animate-spin')} />
            {t('dataLabTopics.refresh')}
          </Button>
        </div>
      </div>

      {/* Summary strip */}
      <div className="flex flex-wrap gap-3 rounded-md border border-border/30 bg-card/30 px-3 py-2 text-[11px]">
        <span><b>{stats.total}</b> {t('dataLabTopics.topicsWord')}</span>
        <span className="text-muted-foreground">·</span>
        <span>parquet: <b>{(stats.byKind['parquet'] ?? 0) + (stats.byKind['external_parquet'] ?? 0)}</b></span>
        <span>sql_table: <b>{stats.byKind['sql_table'] ?? 0}</b></span>
        <span>memory: <b>{stats.byKind['memory'] ?? 0}</b></span>
        <span className="text-muted-foreground">·</span>
        <span><b>{stats.totalEvents.toLocaleString()}</b> {t('dataLabTopics.eventsRecorded')}</span>
        <span><b>{(stats.totalBytes / (1024 * 1024)).toFixed(1)}</b> {t('dataLabTopics.mbOnDisk')}</span>
      </div>

      {/* Global rotation controls */}
      <BusRotationControls />


      {/* Topic list */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-3">
        <div className="rounded-md border border-border/30 bg-card/20 overflow-hidden flex flex-col">
          <ScrollArea className="flex-1 min-h-0">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-card/95 backdrop-blur border-b border-border/30 text-left">
                <tr>
                  <th className="py-2 px-3 font-medium">{t('dataLabTopics.colTopic')}</th>
                  <th className="py-2 px-2 font-medium">{t('dataLabTopics.colStorage')}</th>
                  <th className="py-2 px-2 font-medium">{t('dataLabTopics.colPubSub')}</th>
                  <th className="py-2 px-2 font-medium text-right">{t('dataLabTopics.colVolume')}</th>
                  <th className="py-2 px-2 font-medium">{t('dataLabTopics.colLastActivity')}</th>
                </tr>
              </thead>
              <tbody>
                {query.isLoading ? (
                  <tr>
                    <td colSpan={5} className="py-12 text-center text-muted-foreground">
                      <Loader2 className="inline h-4 w-4 animate-spin" /> {t('dataLabTopics.loading')}
                    </td>
                  </tr>
                ) : topics.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-12 text-center text-muted-foreground">
                      {t('dataLabTopics.noTopics')}
                    </td>
                  </tr>
                ) : (
                  topics.map((topic: TopicSpec) => (
                    <tr
                      key={topic.slug}
                      onClick={() => setSelected(topic)}
                      className={cn(
                        'border-b border-border/10 hover:bg-card/60 cursor-pointer',
                        selected?.slug === topic.slug && 'bg-violet-500/5',
                      )}
                    >
                      <td className="py-2 px-3 align-top">
                        <div className="font-mono text-[11px]">{topic.slug}</div>
                        <div className="text-[10px] text-muted-foreground line-clamp-1">{topic.title}</div>
                      </td>
                      <td className="py-2 px-2 align-top">
                        <Badge
                          style={{ backgroundColor: topicStorageBadgeColor(topic.storage_kind) + '22', color: topicStorageBadgeColor(topic.storage_kind) }}
                          className="text-[10px] font-mono"
                        >
                          {topic.storage_kind}
                        </Badge>
                        {!topic.enabled && (
                          <Badge variant="outline" className="ml-1 text-[9px]">{t('dataLabTopics.badgeDisabled')}</Badge>
                        )}
                        {!topic.is_replayable && (
                          <Badge variant="outline" className="ml-1 text-[9px]">{t('dataLabTopics.badgeLiveOnly')}</Badge>
                        )}
                      </td>
                      <td className="py-2 px-2 align-top text-[10px]">
                        <div className="text-muted-foreground">
                          → {(topic.publishers ?? []).join(', ') || '—'}
                        </div>
                        <div className="text-muted-foreground">
                          ← {(topic.subscribers ?? []).slice(0, 3).join(', ') || '—'}
                          {topic.subscribers && topic.subscribers.length > 3 ? ` (+${topic.subscribers.length - 3})` : ''}
                        </div>
                      </td>
                      <td className="py-2 px-2 align-top text-right font-mono text-[10px]">
                        {formatTopicVolume(topic, t)}
                      </td>
                      <td className="py-2 px-2 align-top text-[10px] text-muted-foreground">
                        {topic.last_published_at ? `${t('dataLabTopics.pubLabel')} ${formatDistanceToNow(new Date(topic.last_published_at))} ${t('dataLabTopics.agoLabel')}` : '—'}
                        {topic.last_replayed_at ? <div>{t('dataLabTopics.repLabel')} {formatDistanceToNow(new Date(topic.last_replayed_at))} {t('dataLabTopics.agoLabel')}</div> : null}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </ScrollArea>
        </div>

        {/* Detail / preview panel */}
        <TopicDetail spec={selected} />
      </div>
    </div>
  )
}

function BusRotationControls() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const settings = useQuery({
    queryKey: ['bus-rotation-settings'],
    queryFn: getBusRotationSettings,
    refetchInterval: 60_000,
  })
  const [globalCapMB, setGlobalCapMB] = useState<string>('')
  useEffect(() => {
    if (settings.data?.global_max_bytes != null) {
      setGlobalCapMB(String(Math.round(settings.data.global_max_bytes / (1024 * 1024))))
    } else {
      setGlobalCapMB('')
    }
  }, [settings.data?.global_max_bytes])

  const updateSettings = useMutation({
    mutationFn: updateBusRotationSettings,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bus-rotation-settings'] })
      qc.invalidateQueries({ queryKey: ['topic-catalog'] })
    },
  })
  const pruneNow = useMutation({
    mutationFn: triggerPruneNow,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bus-rotation-settings'] })
      qc.invalidateQueries({ queryKey: ['topic-catalog'] })
    },
  })

  if (!settings.data) {
    return <div className="text-[10px] text-muted-foreground">{t('dataLabTopics.rotationLoading')}</div>
  }
  const s = settings.data
  const totalMB = s.total_bytes_on_disk / (1024 * 1024)
  const overCap = s.global_max_bytes != null && s.total_bytes_on_disk > s.global_max_bytes

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px]">
      <div className="flex items-center gap-2">
        <HardDrive className="h-3.5 w-3.5 text-amber-700 dark:text-amber-300" />
        <span className="font-medium">{t('dataLabTopics.rotationPolicy')}</span>
      </div>
      <span className="text-muted-foreground">·</span>
      <label className="flex items-center gap-1.5">
        <Power className="h-3 w-3" />
        <input
          type="checkbox"
          checked={s.pruner_enabled}
          onChange={(e) => updateSettings.mutate({ pruner_enabled: e.target.checked })}
          className="h-3 w-3"
        />
        <span>{t('dataLabTopics.pruner')} {s.pruner_enabled ? 'ON' : 'OFF'}</span>
      </label>
      <span className="text-muted-foreground">·</span>
      <div className="flex items-center gap-1.5">
        <span>{t('dataLabTopics.globalCap')}:</span>
        <Input
          value={globalCapMB}
          onChange={(e) => setGlobalCapMB(e.target.value.replace(/[^0-9]/g, ''))}
          placeholder={t('dataLabTopics.capNone')}
          className="h-6 w-24 text-[11px] py-0 px-2"
        />
        <Button
          size="sm" variant="outline" className="h-6 text-[10px] px-2"
          onClick={() => {
            const mb = parseInt(globalCapMB || '0', 10)
            updateSettings.mutate({ global_max_bytes: mb > 0 ? mb * 1024 * 1024 : 0 })
          }}
        >
          {t('dataLabTopics.set')}
        </Button>
      </div>
      <span className="text-muted-foreground">·</span>
      <span className={cn('font-mono', overCap && 'text-red-600 font-semibold')}>
        {t('dataLabTopics.usedLabel')} {totalMB.toFixed(1)} MB
        {s.global_max_bytes != null && ` / ${(s.global_max_bytes / (1024 * 1024)).toFixed(0)} MB`}
      </span>
      <span className="text-muted-foreground">·</span>
      <Button
        size="sm" variant="outline" className="h-6 gap-1 text-[10px]"
        disabled={pruneNow.isPending}
        onClick={() => pruneNow.mutate()}
      >
        {pruneNow.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
        {t('dataLabTopics.pruneNow')}
      </Button>
      {pruneNow.data && (
        <span className="text-[10px] text-muted-foreground">
          {t('dataLabTopics.lastRunFreed', { mb: ((((pruneNow.data.freed_bytes_age || 0) + (pruneNow.data.freed_bytes_per_topic || 0) + (pruneNow.data.freed_bytes_global || 0)) / (1024 * 1024)).toFixed(1)) })}
        </span>
      )}
    </div>
  )
}

function TopicDetail({ spec }: { spec: TopicSpec | null }) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [previewing, setPreviewing] = useState(false)
  const [preview, setPreview] = useState<{ topic: string; samples: any[]; n: number } | null>(null)
  const [retentionDays, setRetentionDays] = useState<string>('')
  const [maxBytesMB, setMaxBytesMB] = useState<string>('')
  useEffect(() => {
    setRetentionDays(spec?.retention_days != null ? String(spec.retention_days) : '')
    setMaxBytesMB(spec?.max_bytes != null ? String(Math.round(spec.max_bytes / (1024 * 1024))) : '')
  }, [spec?.slug, spec?.retention_days, spec?.max_bytes])

  const patch = useMutation({
    mutationFn: (req: Parameters<typeof patchTopic>[1]) => patchTopic(spec!.slug, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['topic-catalog'] })
      qc.invalidateQueries({ queryKey: ['bus-rotation-settings'] })
    },
  })

  if (!spec) {
    return (
      <div className="rounded-md border border-border/30 bg-card/20 p-4 text-[11px] text-muted-foreground">
        {t('dataLabTopics.pickTopic')}
      </div>
    )
  }

  const onPreview = async () => {
    setPreviewing(true)
    setPreview(null)
    try {
      const end_us = Date.now() * 1000
      const start_us = end_us - 24 * 3600 * 1_000_000
      const r = await replayPreview({
        topics: [spec.slug],
        start_us,
        end_us,
        limit: 25,
      })
      setPreview({ topic: spec.slug, samples: r.samples, n: r.n_seen })
    } catch (e: any) {
      setPreview({ topic: spec.slug, samples: [], n: 0 })
    } finally {
      setPreviewing(false)
    }
  }

  return (
    <div className="rounded-md border border-border/30 bg-card/20 flex flex-col min-h-0">
      <div className="border-b border-border/30 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="font-mono text-[12px] font-semibold">{spec.slug}</div>
          <Badge
            style={{ backgroundColor: topicStorageBadgeColor(spec.storage_kind) + '22', color: topicStorageBadgeColor(spec.storage_kind) }}
            className="text-[10px] font-mono"
          >
            {spec.storage_kind}
          </Badge>
        </div>
        <div className="mt-1 text-[11px] text-muted-foreground">{spec.title}</div>
        {spec.description && (
          <div className="mt-2 text-[11px] leading-relaxed">{spec.description}</div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 border-b border-border/30 p-3 text-[11px]">
        <div>
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.storageUri')}</div>
          <div className="font-mono text-[10px] break-all">{spec.storage_uri || '—'}</div>
        </div>
        <div>
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.schemaVersion')}</div>
          <div className="font-mono">v{spec.schema_version}</div>
        </div>
        <div>
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.retentionLabel')}</div>
          <div className="font-mono">{spec.retention_days != null ? t('dataLabTopics.retentionDaysValue', { n: spec.retention_days }) : t('dataLabTopics.forever')}</div>
        </div>
        <div>
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.cumulativeVolume')}</div>
          <div className="font-mono">{formatTopicVolume(spec, t)}</div>
        </div>
        <div className="col-span-2">
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.publishersLabel')}</div>
          <div className="text-[10px]">{spec.publishers.join(', ') || '—'}</div>
        </div>
        <div className="col-span-2">
          <div className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.subscribersLabel')}</div>
          <div className="text-[10px]">{spec.subscribers.join(', ') || '—'}</div>
        </div>
      </div>

      {/* Per-topic controls — only meaningful for storage-kinds we
          actually prune (parquet, external_parquet).  sql_table topics
          surface read-only state. */}
      {(spec.storage_kind === 'parquet' || spec.storage_kind === 'external_parquet') && (
        <div className="border-b border-border/30 p-3 space-y-2 text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-[9px] uppercase text-muted-foreground">{t('dataLabTopics.topicControls')}</span>
            <label className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={spec.enabled}
                onChange={(e) => patch.mutate({ enabled: e.target.checked })}
                className="h-3 w-3"
              />
              <span>{spec.enabled ? t('dataLabTopics.recordingOn') : t('dataLabTopics.recordingOff')}</span>
            </label>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[9px] uppercase text-muted-foreground mb-1">{t('dataLabTopics.retentionDaysLabel')}</div>
              <div className="flex gap-1">
                <Input
                  value={retentionDays}
                  onChange={(e) => setRetentionDays(e.target.value.replace(/[^0-9]/g, ''))}
                  placeholder={t('dataLabTopics.forever')}
                  className="h-6 text-[11px] py-0 px-2"
                />
                <Button
                  size="sm" variant="outline" className="h-6 text-[10px] px-2"
                  onClick={() => patch.mutate({ retention_days: parseInt(retentionDays || '0', 10) })}
                >{t('dataLabTopics.set')}</Button>
              </div>
            </div>
            <div>
              <div className="text-[9px] uppercase text-muted-foreground mb-1">{t('dataLabTopics.maxSizeLabel')}</div>
              <div className="flex gap-1">
                <Input
                  value={maxBytesMB}
                  onChange={(e) => setMaxBytesMB(e.target.value.replace(/[^0-9]/g, ''))}
                  placeholder={t('dataLabTopics.unlimited')}
                  className="h-6 text-[11px] py-0 px-2"
                />
                <Button
                  size="sm" variant="outline" className="h-6 text-[10px] px-2"
                  onClick={() => {
                    const mb = parseInt(maxBytesMB || '0', 10)
                    patch.mutate({ max_bytes: mb > 0 ? mb * 1024 * 1024 : 0 })
                  }}
                >{t('dataLabTopics.set')}</Button>
              </div>
            </div>
          </div>
          {patch.isError && (
            <div className="text-[10px] text-red-600">{t('dataLabTopics.patchFailed')} {String((patch.error as any)?.message || patch.error)}</div>
          )}
        </div>
      )}

      <div className="p-3">
        <Button
          size="sm"
          variant="outline"
          className="w-full gap-1 text-[11px]"
          disabled={previewing}
          onClick={onPreview}
        >
          {previewing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
          {t('dataLabTopics.replayPreview')}
        </Button>
        {preview && (
          <div className="mt-3 max-h-[280px] overflow-auto rounded border border-border/30 bg-card/30 p-2">
            <div className="text-[10px] text-muted-foreground mb-1">
              {t('dataLabTopics.eventsSeen', { n: preview.n, m: preview.samples.length })}
            </div>
            {preview.samples.length === 0 ? (
              <div className="text-[10px] text-muted-foreground">{t('dataLabTopics.noEnvelopes')}</div>
            ) : (
              preview.samples.map((s, i) => (
                <div key={i} className="border-b border-border/10 py-1.5 text-[10px]">
                  <div className="font-mono text-muted-foreground">
                    {s.entity_id.slice(0, 32)}{s.entity_id.length > 32 ? '…' : ''} · t={s.observed_at_us}
                  </div>
                  <div className="font-mono whitespace-pre-wrap break-all leading-snug">
                    {JSON.stringify(s.payload)}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}
