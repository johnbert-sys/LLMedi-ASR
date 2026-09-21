import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  commands,
  events,
  type LastContextUsage,
  type VocabularySummary,
} from "@/bindings";

interface VocabularyOverviewProps {
  /** Bumped by the parent whenever activation, tiers or words change. */
  version: number;
}

/**
 * What the context step can draw on, and what the last dictation actually did
 * with it.
 *
 * The first number is the **candidate pool**, not a limit: what reaches the
 * model is a much smaller selection, decided by the term budget and the
 * model's token window. Calling it a ceiling would be wrong — with every
 * dictionary switched on it runs to thousands of terms while roughly a hundred
 * are ever sent. The line about the last dictation is therefore the honest
 * part: it is measured, not promised.
 *
 * Deliberately narrow: the post-correction pool is not shown here at all.
 */
export const VocabularyOverview: React.FC<VocabularyOverviewProps> = ({
  version,
}) => {
  const { t } = useTranslation();
  const [summary, setSummary] = useState<VocabularySummary | null>(null);
  const [lastUsage, setLastUsage] = useState<LastContextUsage | null>(null);
  const [reloads, setReloads] = useState(0);

  const refresh = useCallback(() => setReloads((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      commands.getVocabularySummary(),
      commands.getLastContextUsage(),
    ]).then(([nextSummary, nextUsage]) => {
      if (!cancelled) {
        setSummary(nextSummary);
        setLastUsage(nextUsage);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [version, reloads]);

  // A finished dictation is the only thing that changes the "last dictation"
  // line, and it happens outside this page — a new history entry is the signal
  // that the transcription pipeline ran to completion.
  useEffect(() => {
    const unlisten = events.historyUpdatePayload.listen((event) => {
      if (event.payload.action === "added") refresh();
    });
    return () => {
      unlisten.then((stop) => stop());
    };
  }, [refresh]);

  if (!summary) return null;

  return (
    <div className="mx-4 rounded-lg border border-mid-gray/20 bg-mid-gray/5 p-4 space-y-2">
      <p className="text-sm">
        {t("dictionary.overview.candidates", {
          count: summary.context_terms,
          dictionaries: summary.active.filter((entry) => entry.context_enabled)
            .length,
        })}
        {summary.context_personal_words > 0 && (
          <>
            {" "}
            {t("dictionary.overview.contextPersonal", {
              count: summary.context_personal_words,
            })}
          </>
        )}
      </p>
      <p className="text-xs text-mid-gray">
        {t("dictionary.overview.candidatesHint")}
      </p>

      {lastUsage && (
        <p className="text-xs">
          <span className="font-medium">
            {t("dictionary.overview.lastRun")}
          </span>{" "}
          {t("dictionary.overview.lastRunSent", {
            sent: lastUsage.sent,
            selected: lastUsage.selected,
          })}
          {lastUsage.context_tokens !== null &&
            lastUsage.context_room !== null && (
              <>
                {" "}
                {t("dictionary.overview.lastRunTokens", {
                  tokens: lastUsage.context_tokens,
                  room: lastUsage.context_room,
                })}
              </>
            )}
          {lastUsage.context_tokens === null && (
            <> {t("dictionary.overview.lastRunUnmeasured")}</>
          )}
          {lastUsage.deferred > 0 && (
            <>
              {" "}
              {t("dictionary.overview.lastRunDeferred", {
                count: lastUsage.deferred,
              })}
            </>
          )}
          {lastUsage.fell_back_without_context && (
            <span className="block mt-1 text-yellow-500">
              {t("dictionary.overview.lastRunFellBack")}
            </span>
          )}
        </p>
      )}
    </div>
  );
};
