import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { open as openFileDialog, save } from "@tauri-apps/plugin-dialog";
import {
  ChevronDown,
  Download,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  commands,
  type DictionaryGroup,
  type DictionaryInfo,
  type DictionaryWord,
} from "@/bindings";
import { useSettings } from "../../../hooks/useSettings";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { VocabularyOverview } from "./VocabularyOverview";

const normalizeWord = (word: string) =>
  word
    .replace(/[<>"']/g, "")
    .replace(/\s+/g, " ")
    .trim();

// Export always writes the plain one-term-per-line format; import also takes a
// spreadsheet export, of which the backend keeps only the first column.
const EXPORT_FILTERS = [{ name: "Text", extensions: ["txt"] }];
const IMPORT_FILTERS = [
  { name: "Wortliste", extensions: ["txt", "csv", "json"] },
];

// Section order on the page: the cross-specialty base first, then what the user
// picks per case (specialty, then anatomy region), with drugs last because they
// are the most opt-in of the four.
const GROUP_ORDER: DictionaryGroup[] = [
  "core",
  "specialty",
  "anatomy",
  "medication",
];

export const DictionarySettings: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { getSetting, updateSetting, isUpdating, refreshSettings } =
    useSettings();
  // The two steps are separate lists: the whole word list is compared against
  // the finished text, while only the selected tier of a context dictionary is
  // ever offered to the model.
  const fuzzyIds = getSetting("active_dictionaries") || [];
  const contextIds = getSetting("context_dictionaries") || [];

  const [dictionaries, setDictionaries] = useState<DictionaryInfo[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [words, setWords] = useState<DictionaryWord[]>([]);
  const [filter, setFilter] = useState("");
  const [newWord, setNewWord] = useState("");
  const [dictFilter, setDictFilter] = useState("");
  const [creating, setCreating] = useState(false);
  const [createName, setCreateName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  /** Which of the two steps the page is showing. Context leads: it is the step
   * that changes what the model writes, so it is the one worth looking at
   * first. */
  const [view, setView] = useState<"fuzzy" | "context">("context");
  /** Whether the "add more dictionaries" disclosure is open. */
  const [showContextPicker, setShowContextPicker] = useState(false);

  // Post-correction: all of them, some, or none. Drives the master switch's
  // three states.
  // Read the live setting, not the snapshot `listDictionaries` returned: a row
  // has to move the moment its switch is flipped, not one refresh later.
  const fuzzyCount = dictionaries.filter((d) => fuzzyIds.includes(d.id)).length;
  const allFuzzyOn =
    dictionaries.length > 0 && fuzzyCount === dictionaries.length;
  const someFuzzyOn = fuzzyCount > 0 && !allFuzzyOn;

  const dictionaryLabel = useCallback(
    (dict: DictionaryInfo) =>
      dict.name ?? t(`dictionary.names.${dict.id}`, { defaultValue: dict.id }),
    [t],
  );

  // Bumped on every change that alters the active vocabulary, so the overview
  // recounts without each handler having to know about it.
  const [overviewVersion, setOverviewVersion] = useState(0);
  // The context list the bulk action replaced, so it can be put back.
  const [contextUndo, setContextUndo] = useState<string[] | null>(null);

  // Several commands change the dictionary lists *in the settings* —
  // `set_fuzzy_all_dictionaries`, creating, importing and deleting. The store
  // does not learn about writes made on the backend side, so pull the settings
  // back in here; otherwise the switches keep showing the state from before the
  // click.
  const refreshDictionaries = useCallback(async () => {
    const [infos] = await Promise.all([
      commands.listDictionaries(),
      refreshSettings(),
    ]);
    setDictionaries(infos);
    setOverviewVersion((v) => v + 1);
  }, [refreshSettings]);

  const refreshWords = useCallback(async (id: string) => {
    const result = await commands.getDictionaryWords(id);
    if (result.status === "ok") {
      setWords(result.data);
    } else {
      toast.error(result.error);
    }
  }, []);

  useEffect(() => {
    refreshDictionaries();
  }, [refreshDictionaries]);

  const activeKey = `${fuzzyIds.join("|")}#${contextIds.join("|")}`;
  useEffect(() => {
    setOverviewVersion((v) => v + 1);
  }, [activeKey]);

  // Drop activated ids that no longer resolve to a dictionary. The backend
  // already skips them when building the model context, so this is tidiness
  // rather than a fix — without it a retired module would sit in the user's
  // settings forever, since toggling another entry preserves the rest of the
  // list. Runs once per visit, and only when there is something to remove.
  const prunedRef = useRef(false);
  useEffect(() => {
    if (prunedRef.current || dictionaries.length === 0) return;
    const known = new Set(dictionaries.map((d) => d.id));
    prunedRef.current = true;
    const liveFuzzy = fuzzyIds.filter((id) => known.has(id));
    if (liveFuzzy.length !== fuzzyIds.length) {
      updateSetting("active_dictionaries", liveFuzzy);
    }
    const liveContext = contextIds.filter((id) => known.has(id));
    if (liveContext.length !== contextIds.length) {
      updateSetting("context_dictionaries", liveContext);
    }
  }, [dictionaries, fuzzyIds, contextIds, updateSetting]);

  const toggleExpanded = async (id: string) => {
    setFilter("");
    setNewWord("");
    setConfirmDeleteId(null);
    setRenamingId(null);
    if (expandedId === id) {
      setExpandedId(null);
      setWords([]);
      return;
    }
    setExpandedId(id);
    setWords([]);
    await refreshWords(id);
  };

  const handleFuzzyToggle = (id: string, enabled: boolean) => {
    updateSetting(
      "active_dictionaries",
      enabled ? [...fuzzyIds, id] : fuzzyIds.filter((x) => x !== id),
    );
  };

  const handleContextToggle = (id: string, enabled: boolean) => {
    setContextUndo(null);
    updateSetting(
      "context_dictionaries",
      enabled ? [...contextIds, id] : contextIds.filter((x) => x !== id),
    );
  };

  // The master switch. On enrols every dictionary, off empties the list. From
  // the "some of them" state a click means "all of them" — the destructive
  // direction should take a deliberate second click.
  const handleFuzzyAll = async () => {
    await commands.setFuzzyAllDictionaries(!allFuzzyOn);
    await refreshDictionaries();
  };

  const handleLevelChange = async (id: string, level: number) => {
    const result = await commands.setDictionaryLevel(id, level);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    await refreshDictionaries();
    // The open editor shows the tier's terms, so it has to follow the switch.
    if (expandedId === id) await refreshWords(id);
  };

  // Bulk action on the context side: take every dictionary out of context
  // selection at once. Post-correction and the tiers are untouched, and the
  // previous list is kept so a single click puts it back.
  const handleNoContextForAll = () => {
    if (contextIds.length === 0) {
      toast.info(t("dictionary.context.bulkNone"));
      return;
    }
    setContextUndo(contextIds);
    updateSetting("context_dictionaries", []);
    toast.success(
      t("dictionary.context.bulkDone", { count: contextIds.length }),
    );
  };

  const handleUndoContext = () => {
    if (!contextUndo) return;
    updateSetting("context_dictionaries", contextUndo);
    setContextUndo(null);
    toast.success(t("dictionary.context.undone"));
  };

  const handleAddWord = async (id: string) => {
    const word = normalizeWord(newWord);
    if (!word || word.length > 100) return;
    if (words.some((w) => w.word === word)) {
      toast.error(t("dictionary.duplicate", { word }));
      return;
    }
    const result = await commands.addDictionaryWord(id, word);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    setNewWord("");
    await Promise.all([refreshWords(id), refreshDictionaries()]);
  };

  const handleRemoveWord = async (id: string, word: string) => {
    const result = await commands.removeDictionaryWord(id, word);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    await Promise.all([refreshWords(id), refreshDictionaries()]);
  };

  const handleReset = async (id: string) => {
    const result = await commands.resetDictionary(id);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    toast.success(t("dictionary.resetDone"));
    await Promise.all([refreshWords(id), refreshDictionaries()]);
  };

  const handleExport = async (dict: DictionaryInfo) => {
    const path = await save({
      defaultPath: `${dictionaryLabel(dict)}.txt`,
      filters: EXPORT_FILTERS,
    });
    if (!path) return;
    const result = await commands.exportDictionary(dict.id, path);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    toast.success(t("dictionary.exportDone"));
  };

  const handleImport = async () => {
    const path = await openFileDialog({
      multiple: false,
      filters: IMPORT_FILTERS,
    });
    if (typeof path !== "string") return;
    const result = await commands.importDictionary(path);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    toast.success(
      t("dictionary.importDone", {
        name: result.data.name ?? result.data.id,
        count: result.data.word_count,
      }),
    );
    await refreshDictionaries();
  };

  const handleCreate = async () => {
    const name = createName.trim();
    if (!name) return;
    const result = await commands.createCustomDictionary(name);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    setCreating(false);
    setCreateName("");
    await refreshDictionaries();
    setExpandedId(result.data.id);
    await refreshWords(result.data.id);
  };

  const handleRename = async (id: string) => {
    const name = renameValue.trim();
    if (!name) return;
    const result = await commands.renameCustomDictionary(id, name);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    setRenamingId(null);
    await refreshDictionaries();
  };

  const handleDelete = async (id: string) => {
    if (confirmDeleteId !== id) {
      setConfirmDeleteId(id);
      return;
    }
    const result = await commands.deleteCustomDictionary(id);
    if (result.status === "error") {
      toast.error(result.error);
      return;
    }
    setConfirmDeleteId(null);
    setExpandedId(null);
    setWords([]);
    await refreshDictionaries();
  };

  // Display order only. The stored order is *priority* order — the backend
  // hands the first MAX_PROMPT_WORDS entries to the model as its prompt — so
  // sorting must not be pushed down into the stored list, or the prompt would
  // silently narrow to whatever starts with "A".
  //
  // Collate through Intl rather than comparing strings: code-point order puts
  // "Ödem" after "Zyanose", which is wrong in every language this ships in.
  const collator = useMemo(
    () => new Intl.Collator(i18n.language, { numeric: true }),
    [i18n.language],
  );

  const visibleWords = useMemo(() => {
    const needle = filter.toLowerCase();
    const matching = needle
      ? words.filter((w) => w.word.toLowerCase().includes(needle))
      : words;
    return [...matching].sort((a, b) => collator.compare(a.word, b.word));
  }, [words, filter, collator]);

  /// The word editor a row opens. The list belongs to the dictionary, not to
  /// one of the two steps, so the same editor appears under post-correction
  /// and under context — and an edit made in one shows up in the other.
  const renderWordEditor = (dict: DictionaryInfo) => (
    <>
      <div className="px-4 pb-4 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <Input
            type="text"
            variant="compact"
            className="max-w-44"
            value={newWord}
            onChange={(e) => setNewWord(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAddWord(dict.id);
              }
            }}
            placeholder={t("dictionary.addPlaceholder")}
          />
          <Button
            variant="primary"
            size="sm"
            onClick={() => handleAddWord(dict.id)}
            disabled={!normalizeWord(newWord)}
          >
            {t("dictionary.add")}
          </Button>
          <div className="flex-1" />
          <Input
            type="text"
            variant="compact"
            className="max-w-40"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={t("dictionary.filterPlaceholder")}
          />
        </div>
        {words.length === 0 ? (
          <p className="text-xs text-mid-gray">{t("dictionary.empty")}</p>
        ) : (
          <div className="flex flex-wrap gap-1 max-h-64 overflow-y-auto">
            {visibleWords.map((entry) => (
              <Button
                key={entry.word}
                onClick={() => handleRemoveWord(dict.id, entry.word)}
                variant="secondary"
                size="sm"
                className="inline-flex items-center gap-1 cursor-pointer"
                aria-label={t("dictionary.removeWord", {
                  word: entry.word,
                })}
              >
                <span className={entry.builtin ? "" : "font-semibold"}>
                  {entry.word}
                </span>
                <X className="w-3 h-3" />
              </Button>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleExport(dict)}
            className="inline-flex items-center gap-1"
          >
            <Download className="w-3.5 h-3.5" />
            {t("dictionary.export")}
          </Button>
          {dict.builtin && dict.modified && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => handleReset(dict.id)}
              className="inline-flex items-center gap-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              {t("dictionary.reset")}
            </Button>
          )}
          {!dict.builtin && (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setRenamingId(dict.id);
                  setRenameValue(dictionaryLabel(dict));
                }}
                className="inline-flex items-center gap-1"
              >
                <Pencil className="w-3.5 h-3.5" />
                {t("dictionary.rename")}
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => handleDelete(dict.id)}
                className="inline-flex items-center gap-1 text-red-500"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {confirmDeleteId === dict.id
                  ? t("dictionary.deleteConfirm")
                  : t("dictionary.delete")}
              </Button>
            </>
          )}
        </div>
      </div>
    </>
  );

  const renderDictionary = (dict: DictionaryInfo) => {
    const expanded = expandedId === dict.id;
    const label = dictionaryLabel(dict);
    const isActive = fuzzyIds.includes(dict.id);
    return (
      <div key={dict.id}>
        <div
          className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-mid-gray/10 transition-colors"
          onClick={() => toggleExpanded(dict.id)}
        >
          <ChevronDown
            className={`w-4 h-4 shrink-0 transition-transform ${
              expanded ? "" : "-rotate-90 rtl:rotate-90"
            }`}
          />
          <div className="flex-1 min-w-0">
            {renamingId === dict.id ? (
              <div
                className="flex items-center gap-2"
                onClick={(e) => e.stopPropagation()}
              >
                <Input
                  type="text"
                  variant="compact"
                  className="max-w-48"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleRename(dict.id);
                    if (e.key === "Escape") setRenamingId(null);
                  }}
                  autoFocus
                />
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleRename(dict.id)}
                  disabled={!renameValue.trim()}
                >
                  {t("dictionary.renameConfirm")}
                </Button>
              </div>
            ) : (
              <p className="text-sm font-medium truncate">{label}</p>
            )}
            <p className="text-xs text-mid-gray">
              {t("dictionary.wordCount", { count: dict.word_count })}
              {dict.modified && (
                <span className="ml-1 rtl:mr-1">
                  · {t("dictionary.modifiedBadge")}
                </span>
              )}
            </p>
          </div>
          <label
            className="flex items-center cursor-pointer"
            onClick={(e) => e.stopPropagation()}
            title={t("dictionary.activeToggle", { name: label })}
          >
            <input
              type="checkbox"
              className="sr-only peer"
              checked={isActive}
              disabled={isUpdating("active_dictionaries")}
              onChange={(e) => handleFuzzyToggle(dict.id, e.target.checked)}
            />
            <div className="relative w-11 h-6 bg-mid-gray/20 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-logo-primary rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-background-ui peer-disabled:opacity-50"></div>
          </label>
        </div>
        {expanded && renderWordEditor(dict)}
      </div>
    );
  };

  /// One row on the context side: a switch, how deep into the list the model
  /// may look, and the same word editor the other step offers — the list is one
  /// list, so it can be edited from wherever the user happens to be.
  const renderContextDictionary = (dict: DictionaryInfo) => {
    const label = dictionaryLabel(dict);
    const enabled = contextIds.includes(dict.id);
    const expanded = expandedId === dict.id;
    return (
      <div key={dict.id}>
        <div
          className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-mid-gray/10 transition-colors"
          onClick={() => toggleExpanded(dict.id)}
        >
          <ChevronDown
            className={`w-4 h-4 shrink-0 transition-transform ${
              expanded ? "" : "-rotate-90 rtl:rotate-90"
            }`}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{label}</p>
            <p className="text-xs text-mid-gray">
              {enabled
                ? t("dictionary.context.offered", {
                    count: dict.context_word_count,
                    total: dict.word_count,
                  })
                : t("dictionary.context.notOffered", {
                    count: dict.word_count,
                  })}
            </p>
          </div>
          {dict.levels.length > 1 && (
            <div
              className={`flex items-center rounded-md border border-mid-gray/30 overflow-hidden transition-opacity ${
                enabled ? "" : "opacity-50"
              }`}
              onClick={(e) => e.stopPropagation()}
              title={t("dictionary.context.levelHint")}
            >
              {dict.levels.map((level) => (
                <button
                  key={level}
                  onClick={() => handleLevelChange(dict.id, level)}
                  className={`px-2 py-0.5 text-xs tabular-nums cursor-pointer transition-colors ${
                    dict.selected_level === level
                      ? "bg-background-ui text-white font-semibold"
                      : "hover:bg-mid-gray/20"
                  }`}
                  aria-pressed={dict.selected_level === level}
                  aria-label={t("dictionary.levelSelect", {
                    name: label,
                    count: level,
                  })}
                >
                  {level}
                </button>
              ))}
            </div>
          )}
          <label
            className="flex items-center cursor-pointer"
            onClick={(e) => e.stopPropagation()}
            title={t("dictionary.context.toggle", { name: label })}
          >
            <input
              type="checkbox"
              className="sr-only peer"
              checked={enabled}
              disabled={isUpdating("context_dictionaries")}
              onChange={(e) => handleContextToggle(dict.id, e.target.checked)}
            />
            <div className="relative w-11 h-6 bg-mid-gray/20 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-logo-primary rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-background-ui peer-disabled:opacity-50"></div>
          </label>
        </div>
        {expanded && (
          <>
            <p className="px-4 pb-2 text-xs text-mid-gray">
              {t("dictionary.context.editHint")}
            </p>
            {renderWordEditor(dict)}
          </>
        )}
      </div>
    );
  };

  // Match on the *displayed* name, which for a bundled module is its
  // translation rather than its id — searching "Herz" must find the module the
  // list calls "Herz und Gefäße".
  const needle = dictFilter.trim().toLowerCase();
  const matchesFilter = (dict: DictionaryInfo) =>
    !needle || dictionaryLabel(dict).toLowerCase().includes(needle);
  const matchCount = dictionaries.filter(matchesFilter).length;
  // With nothing in the context yet the picker is the only thing to do here,
  // so it starts open; a search opens it too, or its hits would stay hidden
  // inside a collapsed section.
  const contextPickerOpen =
    showContextPicker ||
    !!needle ||
    dictionaries.every((d) => !contextIds.includes(d.id));

  /// The grouped sections both views use, so a dictionary sits under the same
  /// heading wherever it appears.
  const renderGroups = (
    entries: DictionaryInfo[],
    render: (dict: DictionaryInfo) => React.ReactNode,
  ) =>
    GROUP_ORDER.map((group) => {
      const inGroup = entries.filter(
        (d) => d.group === group && matchesFilter(d),
      );
      if (inGroup.length === 0) return null;
      return (
        <SettingsGroup
          key={group}
          title={t(`dictionary.groups.${group}`)}
          description={t(`dictionary.groupHints.${group}`, {
            defaultValue: "",
          })}
        >
          {inGroup.map(render)}
        </SettingsGroup>
      );
    });

  /// The context page leads with what is actually being sent. Everything else
  /// sits behind one disclosure: showing 35 switched-off dictionaries above the
  /// two that matter is what made this page confusing.
  const renderContextView = () => {
    const inContext = dictionaries.filter((d) => contextIds.includes(d.id));
    const rest = dictionaries.filter((d) => !contextIds.includes(d.id));
    const restMatching = rest.filter(matchesFilter).length;
    return (
      <div className="space-y-6">
        <VocabularyOverview version={overviewVersion} />

        <div className="space-y-2">
          <p className="px-4 text-sm font-semibold">
            {t("dictionary.context.activeHeading", { count: inContext.length })}
          </p>
          {inContext.length === 0 ? (
            <p className="px-4 text-xs text-mid-gray">
              {t("dictionary.context.emptyActive")}
            </p>
          ) : (
            renderGroups(inContext, renderContextDictionary)
          )}
        </div>

        <div className="space-y-2">
          <button
            onClick={() => setShowContextPicker((open) => !open)}
            aria-expanded={contextPickerOpen}
            className="flex items-center gap-2 px-4 text-sm cursor-pointer hover:text-logo-primary transition-colors"
          >
            <ChevronDown
              className={`w-4 h-4 shrink-0 transition-transform ${
                contextPickerOpen ? "" : "-rotate-90 rtl:rotate-90"
              }`}
            />
            {t("dictionary.context.addHeading", { count: rest.length })}
          </button>
          {contextPickerOpen && (
            <>
              {needle && restMatching === 0 && (
                <p className="px-4 text-xs text-mid-gray">
                  {t("dictionary.searchEmpty", { query: dictFilter.trim() })}
                </p>
              )}
              {renderGroups(rest, renderContextDictionary)}
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="px-4 space-y-2">
        <div
          className="flex items-center rounded-md border border-mid-gray/30 overflow-hidden w-fit"
          role="tablist"
        >
          {(
            [
              ["context", t("dictionary.context.tab")],
              ["fuzzy", t("dictionary.fuzzy.tab")],
            ] as ["fuzzy" | "context", string][]
          ).map(([id, title]) => (
            <button
              key={id}
              role="tab"
              aria-selected={view === id}
              onClick={() => {
                // An editor left open in the other tab would reappear in a
                // place the user did not open it.
                setView(id);
                setExpandedId(null);
                setWords([]);
              }}
              className={`px-3 py-1 text-sm cursor-pointer transition-colors ${
                view === id
                  ? "bg-background-ui text-white font-semibold"
                  : "hover:bg-mid-gray/20"
              }`}
            >
              {title}
            </button>
          ))}
        </div>
        <p className="text-xs text-mid-gray">
          {view === "fuzzy"
            ? t("dictionary.fuzzy.intro")
            : t("dictionary.context.intro")}
        </p>
      </div>
      <div className="px-4">
        <div className="relative">
          <Search className="w-4 h-4 absolute start-2 top-1/2 -translate-y-1/2 text-mid-gray pointer-events-none" />
          <Input
            type="search"
            variant="compact"
            className="w-full ps-8"
            value={dictFilter}
            onChange={(e) => setDictFilter(e.target.value)}
            placeholder={t("dictionary.searchPlaceholder")}
            aria-label={t("dictionary.searchPlaceholder")}
          />
        </div>
        {needle && matchCount === 0 && (
          <p className="text-xs text-mid-gray mt-2">
            {t("dictionary.searchEmpty", { query: dictFilter.trim() })}
          </p>
        )}
      </div>

      {view === "fuzzy" ? (
        <div className="px-4 space-y-2">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={allFuzzyOn}
              ref={(node) => {
                // "Some of them" is a real third state, and only the DOM can
                // show it — React has no prop for it.
                if (node) node.indeterminate = someFuzzyOn;
              }}
              onChange={handleFuzzyAll}
            />
            <div
              className={`relative w-11 h-6 shrink-0 rounded-full peer peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-logo-primary peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-background-ui ${
                someFuzzyOn
                  ? "bg-background-ui/40 after:translate-x-1/2"
                  : "bg-mid-gray/20"
              }`}
            ></div>
            <span className="text-sm font-medium">
              {t("dictionary.fuzzy.allSwitch")}
            </span>
            <span className="text-xs text-mid-gray tabular-nums">
              {t("dictionary.fuzzy.allSwitchCount", {
                active: fuzzyCount,
                total: dictionaries.length,
              })}
            </span>
          </label>
          <p className="text-xs text-mid-gray">
            {t("dictionary.fuzzy.allSwitchHint")}
          </p>
        </div>
      ) : (
        <div className="px-4 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleNoContextForAll}
            >
              {t("dictionary.context.bulkOff")}
            </Button>
            {contextUndo && contextUndo.length > 0 && (
              <Button variant="secondary" size="sm" onClick={handleUndoContext}>
                {t("dictionary.context.undo")}
              </Button>
            )}
          </div>
          <p className="text-xs text-mid-gray">
            {t("dictionary.context.personalNote")}
          </p>
        </div>
      )}

      {view === "fuzzy"
        ? renderGroups(dictionaries, renderDictionary)
        : renderContextView()}

      {view === "fuzzy" && (
        <div className="flex items-center gap-2 flex-wrap px-4">
          {creating ? (
            <>
              <Input
                type="text"
                variant="compact"
                className="max-w-56"
                value={createName}
                onChange={(e) => setCreateName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreate();
                  if (e.key === "Escape") setCreating(false);
                }}
                placeholder={t("dictionary.createNamePlaceholder")}
                autoFocus
              />
              <Button
                variant="primary"
                size="sm"
                onClick={handleCreate}
                disabled={!createName.trim()}
              >
                {t("dictionary.createConfirm")}
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setCreating(false)}
              >
                {t("dictionary.cancel")}
              </Button>
            </>
          ) : (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setCreating(true)}
              className="inline-flex items-center gap-1"
            >
              <Plus className="w-3.5 h-3.5" />
              {t("dictionary.create")}
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleImport}
            className="inline-flex items-center gap-1"
          >
            <Upload className="w-3.5 h-3.5" />
            {t("dictionary.import")}
          </Button>
          <span className="text-xs text-mid-gray">
            {t("dictionary.importHint")}
          </span>
        </div>
      )}
    </div>
  );
};
