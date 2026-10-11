import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, PlusCircle, Shield } from 'lucide-react'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { createSimulationAccount } from '../services/api'

interface CreateSandboxAccountDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated?: (accountId: string) => void
}

const MIN_INITIAL_CAPITAL = 100
const MAX_INITIAL_CAPITAL = 10_000_000

function errorMessage(error: unknown, t: TFunction): string {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  const message = (error as any)?.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  return t('createSandboxAccountDialog.failedToCreate')
}

export default function CreateSandboxAccountDialog({
  open,
  onOpenChange,
  onCreated,
}: CreateSandboxAccountDialogProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [initialCapital, setInitialCapital] = useState('10000')

  useEffect(() => {
    if (!open) return
    setName('')
    setInitialCapital('10000')
  }, [open])

  const trimmedName = name.trim()
  const capitalValue = Number(initialCapital)
  const nameValid = trimmedName.length >= 1 && trimmedName.length <= 100
  const capitalValid =
    Number.isFinite(capitalValue) && capitalValue >= MIN_INITIAL_CAPITAL && capitalValue <= MAX_INITIAL_CAPITAL
  const canSubmit = nameValid && capitalValid

  const createMutation = useMutation({
    mutationFn: () =>
      createSimulationAccount({
        name: trimmedName,
        initial_capital: capitalValue,
      }),
    onSuccess: async (result) => {
      // Await the refetch before selecting the new account: setting the
      // selection while AccountModeSelector's cached account list still
      // lacks the new id trips its stale-id reset, which would clobber
      // the selection back to the first account.
      await queryClient.invalidateQueries({ queryKey: ['simulation-accounts'] })
      onCreated?.(result.account_id)
      onOpenChange(false)
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[440px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            <Shield className="h-4 w-4 text-amber-400" />
            {t('createSandboxAccountDialog.title')}
          </DialogTitle>
          <DialogDescription>
            {t('createSandboxAccountDialog.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="space-y-1">
            <Label>{t('createSandboxAccountDialog.name')}</Label>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder={t('createSandboxAccountDialog.namePlaceholder')}
              maxLength={100}
            />
          </div>
          <div className="space-y-1">
            <Label>{t('createSandboxAccountDialog.initialCapital')}</Label>
            <Input
              value={initialCapital}
              onChange={(event) => setInitialCapital(event.target.value)}
              inputMode="decimal"
              placeholder="10000"
            />
            <p className="text-[11px] text-muted-foreground">
              {t('createSandboxAccountDialog.capitalHint', {
                min: MIN_INITIAL_CAPITAL.toLocaleString(),
                max: MAX_INITIAL_CAPITAL.toLocaleString(),
              })}
            </p>
          </div>

          {createMutation.isError && (
            <p className="text-xs text-rose-400">{errorMessage(createMutation.error, t)}</p>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={createMutation.isPending}>
            {t('common.cancel')}
          </Button>
          <Button
            onClick={() => createMutation.mutate()}
            disabled={!canSubmit || createMutation.isPending}
            className="gap-2"
          >
            {createMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlusCircle className="h-4 w-4" />}
            {t('createSandboxAccountDialog.create')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
